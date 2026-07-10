#!/usr/bin/env python3
"""Measure auditable expert-importance signals before an MoE pruning sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker  # noqa: E402
from ultralytics.nn.modules.moe.schedule import usage_gini  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for an artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_images(data: Path, split: str) -> tuple[dict[str, Any], list[Path]]:
    """Resolve image files for a dataset split without constructing a trainer dataset."""
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    if split not in config:
        raise ValueError(f"dataset YAML has no {split!r} split: {data}")
    root = Path(config.get("path") or data.parent)
    root = root if root.is_absolute() else (data.parent / root).resolve()
    sources = config[split] if isinstance(config[split], list) else [config[split]]
    images: list[Path] = []
    for source in sources:
        path = Path(source)
        path = path if path.is_absolute() else root / path
        if path.is_dir():
            images.extend(item.absolute() for item in path.rglob("*.*") if item.suffix.lower() in IMAGE_SUFFIXES)
        elif path.suffix.lower() == ".txt":
            for raw in path.read_text(encoding="utf-8").splitlines():
                item = Path(raw.strip())
                if not item.is_absolute():
                    item = path.parent / item
                if item.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(item.absolute())
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path.absolute())
        else:
            raise ValueError(f"unsupported dataset split source: {path}")
    return config, sorted(set(images))


def prepare_fraction_dataset(data: Path, split: str, fraction: float, seed: int, output: Path) -> tuple[Path, int, int]:
    """Materialize a deterministic image list because validation ignores the generic fraction option."""
    config, images = _split_images(data, split)
    if not images:
        raise ValueError(f"no images found for split={split!r} in {data}")
    total = len(images)
    selected_count = max(1, round(total * fraction))
    if selected_count == total:
        return data, total, total

    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    selected = sorted(shuffled[:selected_count])
    output.parent.mkdir(parents=True, exist_ok=True)
    image_list = output.with_suffix(".images.txt")
    subset_yaml = output.with_suffix(".dataset.yaml")
    image_list.write_text("".join(f"{path}\n" for path in selected), encoding="utf-8")
    config[split] = str(image_list.resolve())
    subset_yaml.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return subset_yaml, total, selected_count


def summarize_usage_stats(
    usage_stats: dict[str, dict[int, Any]],
    num_experts_by_layer: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Summarize hard-hit and normalized gate-mass signals per router layer."""
    summaries: dict[str, dict[str, Any]] = {}
    num_experts_by_layer = num_experts_by_layer or {}

    for layer_name in sorted(set(usage_stats) | set(num_experts_by_layer)):
        stats = usage_stats.get(layer_name, {})
        observed_ids = set(stats)
        expert_count = int(num_experts_by_layer.get(layer_name, max(observed_ids, default=-1) + 1))
        expert_ids = range(max(expert_count, max(observed_ids, default=-1) + 1))
        total_hits = sum(float(item.hits) for item in stats.values())
        total_weight = sum(float(item.weighted_sum) for item in stats.values())
        experts = []
        hard_usage = []
        soft_contribution = []

        for expert_id in expert_ids:
            item = stats.get(expert_id)
            hits = float(item.hits) if item is not None else 0.0
            weighted_sum = float(item.weighted_sum) if item is not None else 0.0
            hard = hits / total_hits if total_hits > 0 else 0.0
            soft = weighted_sum / total_weight if total_weight > 0 else 0.0
            average_weight = weighted_sum / hits if hits > 0 else 0.0
            hard_usage.append(hard)
            soft_contribution.append(soft)
            experts.append(
                {
                    "expert_id": expert_id,
                    "hits": hits,
                    "hard_usage": hard,
                    "average_gate_weight": average_weight,
                    "soft_contribution": soft,
                }
            )

        summaries[layer_name] = {
            "expert_count": len(experts),
            "total_hits": total_hits,
            "total_gate_mass": total_weight,
            "hard_usage_gini": usage_gini(hard_usage),
            "soft_contribution_gini": usage_gini(soft_contribution),
            "experts": experts,
        }
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--fraction", type=float, default=1.0, help="Deterministic dataset fraction used by validation."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used when selecting a dataset fraction.")
    parser.add_argument("--moe-inference-mode", choices=("dense", "sparse"), default="dense")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit("--fraction must be in (0, 1]")
    validation_data, available_images, selected_images = prepare_fraction_dataset(
        args.data, args.split, args.fraction, args.seed, args.output
    )
    model = YOLO(str(args.model))
    configured_moe_layers = 0
    for module in model.model.modules():
        if hasattr(module, "use_sparse_inference"):
            module.use_sparse_inference = args.moe_inference_mode == "sparse"
            configured_moe_layers += 1
    router_experts = {
        name: int(module.num_experts)
        for name, module in model.model.named_modules()
        if hasattr(module, "num_experts") and (name.endswith("routing") or name.endswith("router"))
    }
    expert_signature = [
        {
            "layer": name,
            "num_experts": int(module.num_experts),
            "top_k": int(getattr(module, "top_k", module.num_experts)),
        }
        for name, module in model.model.named_modules()
        if hasattr(module, "experts") and hasattr(module, "routing") and hasattr(module, "num_experts")
    ]
    named_parameters = list(model.model.named_parameters())
    lora_parameters = [(name, parameter) for name, parameter in named_parameters if "lora" in name.lower()]
    adapter_module_count = sum(
        1
        for module in model.model.modules()
        if "lora" in type(module).__name__.lower() or hasattr(module, "lora_A") or hasattr(module, "lora_B")
    )
    observed: dict[str, int] = {}

    def capture_validation_extent(validator: Any) -> None:
        observed["images"] = int(validator.seen)
        observed["batches"] = len(validator.dataloader)

    model.add_callback("on_val_end", capture_validation_extent)

    with ExpertUsageTracker(model.model) as tracker:
        metrics = model.val(
            data=str(validation_data),
            split=args.split,
            device=args.device,
            batch=args.batch,
            imgsz=args.imgsz,
            verbose=False,
            plots=False,
            fraction=1.0,
            seed=args.seed,
        )
    layers = summarize_usage_stats(tracker.usage_stats, router_experts)
    nontrivial_layers = sum(
        1
        for layer in layers.values()
        if len({round(expert["soft_contribution"], 12) for expert in layer["experts"]}) > 1
    )
    box = getattr(metrics, "box", None)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "revision": revision,
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "data": str(args.data),
        "validation_data": str(validation_data),
        "split": args.split,
        "requested_fraction": args.fraction,
        "available_images": available_images,
        "selected_images": selected_images,
        "observed_images": observed.get("images"),
        "observed_batches": observed.get("batches"),
        "effective_fraction": observed.get("images", 0) / available_images,
        "seed": args.seed,
        "device": args.device,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "moe_inference_mode": args.moe_inference_mode,
        "configured_moe_layers": configured_moe_layers,
        "params": sum(parameter.numel() for parameter in model.model.parameters()),
        "gflops": get_flops(model.model, imgsz=args.imgsz),
        "trainable_params": sum(parameter.numel() for _, parameter in named_parameters if parameter.requires_grad),
        "expert_signature": expert_signature,
        "lora": {
            "enabled": bool(getattr(model.model, "lora_enabled", False)),
            "adapter_module_count": adapter_module_count,
            "parameter_count": sum(parameter.numel() for _, parameter in lora_parameters),
            "trainable_parameter_count": sum(
                parameter.numel() for _, parameter in lora_parameters if parameter.requires_grad
            ),
        },
        "metrics": {
            "mAP50-95": float(box.map) if box is not None else None,
            "mAP50": float(box.map50) if box is not None else None,
            "speed_ms_per_image": dict(getattr(metrics, "speed", {}) or {}),
        },
        "opportunity_count": 1,
        "event_count": int(bool(layers)),
        "layer_event_count": len(layers),
        "nontrivial_signal_layer_count": nontrivial_layers,
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[moe-signal] wrote {args.output}")
    print(
        f"[moe-signal] layers={len(layers)} nontrivial={nontrivial_layers} "
        f"mAP50-95={payload['metrics']['mAP50-95']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
