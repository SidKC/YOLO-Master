#!/usr/bin/env python3
"""Run the B1 real DetectionModel integration smoke on one visible CUDA device."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MODEL_CFG = ROOT / "ultralytics/cfg/models/master/v0_15/det/yolo-master-b1-tiny.yaml"
DEFAULT_OUTPUT = ROOT / "runs/b1_a100_detector_smoke/result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="Visible CUDA device, default: cuda:0")
    parser.add_argument("--model-cfg", type=Path, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parameter_digest(named_parameters: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def finite_nonzero_gradients(parameters: list[Any]) -> bool:
    import torch

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients) and any(
        float(gradient.detach().abs().max()) > 0.0 for gradient in gradients
    )


def max_parameter_delta(before: list[Any], parameters: list[Any]) -> float:
    import torch

    deltas = [(parameter.detach() - initial).abs().max() for initial, parameter in zip(before, parameters)]
    if not deltas or not all(bool(torch.isfinite(delta)) for delta in deltas):
        raise AssertionError("parameter delta is missing or non-finite")
    return max(float(delta) for delta in deltas)


def configure_router(module: Any) -> None:
    """Set deterministic routes so both experts can be checked independently."""
    import torch

    with torch.no_grad():
        module.condition_projection.weight.zero_()
        module.condition_projection.bias.zero_()
        module.condition_projection.weight[0, 0] = 1.0
        module.router.weight.zero_()
        module.router.bias.zero_()
        module.router.weight[0, module.hidden_dim] = 1.0
        module.router.weight[1, module.hidden_dim] = -1.0


def make_batch(torch: Any, images: Any, condition: Any) -> dict[str, Any]:
    batch_size = images.shape[0]
    return {
        "img": images,
        "text_condition": condition,
        "batch_idx": torch.arange(batch_size, device=images.device, dtype=torch.long),
        "cls": torch.zeros(batch_size, 1, device=images.device),
        "bboxes": torch.full((batch_size, 4), 0.5, device=images.device),
    }


def run_case(
    model_cfg: Path,
    device: Any,
    batch_size: int,
    expected_experts: list[int],
    seed: int,
) -> dict[str, Any]:
    import torch

    from ultralytics.nn.mixture_loss import CompositeCriterion
    from ultralytics.nn.modules import TextConditionedMoT
    from ultralytics.nn.modules.routing_protocol import clear_aux_records, collect_aux_loss, current_aux_step
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = DetectionModel(model_cfg, ch=3, nc=2, verbose=False).to(device).train()
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    modules = [module for module in model.modules() if isinstance(module, TextConditionedMoT)]
    if len(modules) != 1:
        raise AssertionError(f"expected one TextConditionedMoT, found {len(modules)}")
    module = modules[0]
    model.requires_grad_(False)
    module.requires_grad_(True)
    configure_router(module)

    module_ids = {id(parameter) for parameter in module.parameters()}
    detector_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if id(parameter) not in module_ids
    ]
    detector_before = parameter_digest(detector_parameters)

    images = torch.rand(batch_size, 3, 64, 64, device=device)
    rows = [[1.0] + [0.0] * 511 if expert == 0 else [-1.0] + [0.0] * 511 for expert in expected_experts]
    condition = torch.tensor(rows, device=device, dtype=torch.float32, requires_grad=True)
    condition_before = condition.detach().clone()

    prediction = model.predict(images, condition=condition)
    if not isinstance(prediction, dict) or "boxes" not in prediction:
        raise AssertionError("detector prediction did not return boxes")
    if not bool(torch.isfinite(prediction["boxes"]).all()):
        raise AssertionError("detector prediction contains non-finite values")
    if module.last_routing_snapshot["executed_expert"] != expected_experts:
        raise AssertionError("prediction path selected unexpected experts")

    clear_aux_records(step=1000 + seed)
    loss, items = model.loss(make_batch(torch, images, condition))
    if not isinstance(model.criterion, CompositeCriterion):
        raise AssertionError("model criterion is not CompositeCriterion")
    if not bool(torch.isfinite(loss).all()) or not loss.requires_grad:
        raise AssertionError("detector loss is non-finite or detached")
    if not isinstance(items, torch.Tensor) or not bool(torch.isfinite(items).all()):
        raise AssertionError("loss items are invalid")
    if module.last_routing_snapshot["executed_expert"] != expected_experts:
        raise AssertionError("loss path selected unexpected experts")

    aux, diagnostics = collect_aux_loss(
        model,
        step=current_aux_step(),
        include_kinds=("mot",),
        return_diagnostics=True,
    )
    if diagnostics["counts_by_kind"].get("mot") != 1:
        raise AssertionError(f"unexpected MoT aux count: {diagnostics}")
    if any(diagnostics[key] != 0 for key in ("stale_skipped", "eval_skipped", "duplicate_skipped")):
        raise AssertionError(f"unexpected auxiliary diagnostics: {diagnostics}")
    if not aux.requires_grad or not bool(torch.isfinite(aux)):
        raise AssertionError("routing auxiliary loss is invalid")

    optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)
    selected = sorted(set(expected_experts))
    expert_parameters = {index: list(module.experts[index].parameters()) for index in selected}
    expert_before = {
        index: [parameter.detach().clone() for parameter in parameters]
        for index, parameters in expert_parameters.items()
    }
    module_before = [parameter.detach().clone() for parameter in module.parameters()]

    optimizer.zero_grad(set_to_none=True)
    loss.sum().backward()
    if condition.grad is not None or not torch.equal(condition.detach(), condition_before):
        raise AssertionError("frozen text condition changed or received a gradient")
    if not finite_nonzero_gradients([module.condition_projection.weight, module.router.weight]):
        raise AssertionError("condition projection or router gradient is missing, zero or non-finite")
    for index, parameters in expert_parameters.items():
        if not finite_nonzero_gradients(parameters):
            raise AssertionError(f"expert {index} gradient is missing, zero or non-finite")

    optimizer.step()
    module_delta = max_parameter_delta(module_before, list(module.parameters()))
    expert_deltas = {
        str(index): max_parameter_delta(expert_before[index], parameters)
        for index, parameters in expert_parameters.items()
    }
    if module_delta <= 0.0 or any(delta <= 0.0 for delta in expert_deltas.values()):
        raise AssertionError("router module or selected expert did not update")
    if parameter_digest(detector_parameters) != detector_before:
        raise AssertionError("a frozen detector parameter changed")

    torch.cuda.synchronize(device)
    return {
        "status": "PASS",
        "seed": seed,
        "batch_size": batch_size,
        "expected_experts": expected_experts,
        "executed_experts": module.last_routing_snapshot["executed_expert"],
        "loss": float(loss.detach().sum()),
        "aux_loss": float(aux.detach()),
        "module_max_parameter_delta": module_delta,
        "expert_max_parameter_delta": expert_deltas,
        "condition_unchanged": True,
        "condition_grad_is_none": True,
        "frozen_detector_unchanged": True,
        "aux_count": diagnostics["counts_by_kind"]["mot"],
        "stale_eval_duplicate_aux_counts": [
            diagnostics["stale_skipped"],
            diagnostics["eval_skipped"],
            diagnostics["duplicate_skipped"],
        ],
    }


def main() -> None:
    args = parse_args()
    model_cfg = args.model_cfg.expanduser().resolve()
    if not model_cfg.is_file():
        raise FileNotFoundError(f"model config not found: {model_cfg}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    properties = torch.cuda.get_device_properties(device)
    started_at = utc_now()
    started_monotonic = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    cases = [
        run_case(model_cfg, device, 1, [0], 20260824),
        run_case(model_cfg, device, 1, [1], 20260825),
        run_case(model_cfg, device, 2, [0, 1], 20260826),
    ]
    torch.cuda.synchronize(device)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "source_commit": git_revision(),
        "model": "real DetectionModel",
        "model_config": str(model_cfg.relative_to(ROOT)) if model_cfg.is_relative_to(ROOT) else str(model_cfg),
        "configuration": {
            "image_size": [64, 64],
            "precision": "FP32",
            "optimizer": "SGD",
            "learning_rate": 0.001,
        },
        "device": {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cases": cases,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "wall_seconds": time.monotonic() - started_monotonic,
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
