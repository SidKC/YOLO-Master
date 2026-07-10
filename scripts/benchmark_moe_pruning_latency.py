#!/usr/bin/env python3
"""Benchmark cold and warm model-forward latency for YOLO/MoE checkpoints.

This script intentionally benchmarks only the PyTorch model forward pass on a
synthetic tensor. Dataset loading, image decoding, preprocessing, NMS, and
serialization are outside the measured region, making pruning variants easier
to compare under identical inputs.

Example:
    python scripts/benchmark_moe_pruning_latency.py \
        --model runs/baseline/weights/best.pt runs/pruned/weights/best.pt \
        --device 1 --batch-size 1 --imgsz 640 --warmup 20 --reps 200 \
        --output-dir runs/issue52_latency
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import platform
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch: Any = None
YOLO: Any = None


def load_runtime() -> None:
    """Import heavy benchmark dependencies only after CLI parsing (so --help works anywhere)."""
    global YOLO, torch
    try:
        torch = importlib.import_module("torch")
        YOLO = importlib.import_module("ultralytics").YOLO
    except ImportError as error:
        raise SystemExit(f"benchmark runtime dependency is unavailable: {error}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, nargs="+", required=True, help="One or more YOLO .pt checkpoints or model YAMLs.")
    parser.add_argument("--device", default="0", help="PyTorch device: CUDA index, cuda:N, cpu, mps, or auto.")
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic input batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input image size in pixels.")
    parser.add_argument("--warmup", type=int, default=20, help="Unmeasured forwards after the cold forward.")
    parser.add_argument("--reps", type=int, default=200, help="Number of measured warm forwards.")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--fuse", action="store_true", help="Fuse supported Conv/BN layers before measurement.")
    parser.add_argument(
        "--moe-inference-mode",
        choices=("dense", "sparse"),
        default="dense",
        help="Force a consistent ES_MOE inference path across checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used to create the synthetic input.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/issue52_latency")
    parser.add_argument("--name", default="moe_pruning_latency", help="Stem for the output JSON and CSV files.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.imgsz < 1:
        raise SystemExit("--imgsz must be at least 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.reps < 1:
        raise SystemExit("--reps must be at least 1")
    if not args.name or Path(args.name).name != args.name:
        raise SystemExit("--name must be a filename stem, not a path")


def resolve_device(value: str) -> torch.device:
    value = value.strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA device requested ({device}), but CUDA is unavailable")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("MPS requested, but MPS is unavailable")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]
    if device.type == "cpu" and dtype == torch.float16:
        raise SystemExit("fp16 model-forward benchmarking is unsupported on CPU; use fp32 or bf16")
    return dtype


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def percentile(values: list[float], q: float) -> float:
    """Return a linear-interpolated percentile for q in [0, 1]."""
    if not values:
        raise ValueError("cannot calculate a percentile from no samples")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def latency_stats(values_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values_ms),
        "p50_ms": percentile(values_ms, 0.50),
        "p95_ms": percentile(values_ms, 0.95),
        "p99_ms": percentile(values_ms, 0.99),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {"type": device.type, "torch_device": str(device)}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": properties.total_memory,
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    elif device.type == "mps":
        metadata["name"] = "Apple MPS"
    else:
        metadata.update(
            {"name": platform.processor() or platform.machine(), "logical_cpu_count": __import__("os").cpu_count()}
        )
    return metadata


def timed_forward(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device) -> float:
    synchronize(device)
    start = time.perf_counter_ns()
    output = model(tensor)
    synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    del output
    return elapsed_ms


def load_model(
    path: Path, device: torch.device, dtype: torch.dtype, fuse: bool, moe_inference_mode: str
) -> tuple[torch.nn.Module, float, int]:
    synchronize(device)
    started = time.perf_counter_ns()
    wrapper = YOLO(str(path))
    model = wrapper.model
    configured_moe_layers = 0
    for module in model.modules():
        if hasattr(module, "use_sparse_inference"):
            module.use_sparse_inference = moe_inference_mode == "sparse"
            configured_moe_layers += 1
    if fuse and hasattr(model, "fuse"):
        model.fuse()
    model = model.to(device=device, dtype=dtype).eval()
    synchronize(device)
    load_ms = (time.perf_counter_ns() - started) / 1e6
    return model, load_ms, configured_moe_layers


def benchmark_model(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"model does not exist: {resolved}")

    model_hash = sha256_file(resolved)
    model, load_ms, configured_moe_layers = load_model(
        resolved, device, dtype, args.fuse, args.moe_inference_mode
    )
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    torch.manual_seed(args.seed)  # Recreate the same synthetic input for every checkpoint.
    tensor = torch.randn(
        args.batch_size,
        3,
        args.imgsz,
        args.imgsz,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        cold_batch_ms = timed_forward(model, tensor, device)
        for _ in range(args.warmup):
            _ = model(tensor)
        synchronize(device)
        warm_batch_ms = [timed_forward(model, tensor, device) for _ in range(args.reps)]

    batch_stats = latency_stats(warm_batch_ms)
    per_image_ms = [value / args.batch_size for value in warm_batch_ms]
    image_stats = latency_stats(per_image_ms)
    throughput_images_s = args.batch_size * 1000.0 / batch_stats["mean_ms"]
    throughput_batches_s = 1000.0 / batch_stats["mean_ms"]

    return {
        "model": str(resolved),
        "model_sha256": model_hash,
        "model_size_bytes": resolved.stat().st_size,
        "parameters": params,
        "trainable_parameters": trainable_params,
        "load_ms": load_ms,
        "configured_moe_layers": configured_moe_layers,
        "cold": {
            "batch_ms": cold_batch_ms,
            "per_image_ms": cold_batch_ms / args.batch_size,
        },
        "warm": {
            "batch_latency": batch_stats,
            "per_image_latency": image_stats,
            "throughput_images_s": throughput_images_s,
            "throughput_batches_s": throughput_batches_s,
            "samples_batch_ms": warm_batch_ms,
        },
    }


def flatten_result(result: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    batch = result["warm"]["batch_latency"]
    image = result["warm"]["per_image_latency"]
    return {
        "model": result["model"],
        "model_sha256": result["model_sha256"],
        "model_size_bytes": result["model_size_bytes"],
        "parameters": result["parameters"],
        "trainable_parameters": result["trainable_parameters"],
        "device": common["device"]["torch_device"],
        "device_name": common["device"]["name"],
        "benchmark_scope": common["scope"],
        "created_at_utc": common["created_at_utc"],
        "git_revision": common["git_revision"],
        "torch_version": common["torch_version"],
        "dtype": common["dtype"],
        "batch_size": common["batch_size"],
        "imgsz": common["imgsz"],
        "warmup": common["warmup"],
        "reps": common["reps"],
        "fused": common["fused"],
        "moe_inference_mode": common["moe_inference_mode"],
        "configured_moe_layers": result["configured_moe_layers"],
        "load_ms": result["load_ms"],
        "cold_batch_ms": result["cold"]["batch_ms"],
        "cold_per_image_ms": result["cold"]["per_image_ms"],
        **{f"warm_batch_{key}": value for key, value in batch.items()},
        **{f"warm_per_image_{key}": value for key, value in image.items()},
        "throughput_images_s": result["warm"]["throughput_images_s"],
        "throughput_batches_s": result["warm"]["throughput_batches_s"],
        "command": common["command"],
    }


def write_outputs(output_dir: Path, name: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    csv_path = output_dir / f"{name}.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = [flatten_result(result, payload["benchmark"]) for result in payload["results"]]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> int:
    args = parse_args()
    validate_args(args)
    load_runtime()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch.manual_seed(args.seed)

    common = {
        "scope": "model_forward_only",
        "excluded": ["data_loading", "decode", "preprocess", "nms", "serialization"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "git_revision": git_revision(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "ultralytics_source": str(ROOT),
        "device": device_metadata(device),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "imgsz": args.imgsz,
        "warmup": args.warmup,
        "reps": args.reps,
        "seed": args.seed,
        "fused": args.fuse,
        "moe_inference_mode": args.moe_inference_mode,
    }
    results = [benchmark_model(path, device, dtype, args) for path in args.model]
    payload = {"benchmark": common, "results": results}
    json_path, csv_path = write_outputs(args.output_dir.expanduser().resolve(), args.name, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[benchmark] JSON: {json_path}")
    print(f"[benchmark] CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
