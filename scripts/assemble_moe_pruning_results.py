#!/usr/bin/env python3
"""Assemble an auditable 5x2 Issue #52 pruning table from measured artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


RECOVERIES = ("direct", "lora10")
MEASUREMENT_FIELDS = (
    "mAP50-95",
    "mAP50",
    "gflops",
    "latency_ms",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_images_s",
    "params_m",
    "experts_per_layer",
    "expert_usage_gini",
    "convergence_epoch_95pct",
    "recovery_structure_status",
    "eval_model_sha256",
    "eval_json",
    "latency_jsons",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--plan-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def canonical_layer_name(name: str) -> str:
    """Normalize direct and PEFT-wrapped YOLO layer paths to ``model.N``."""
    normalized = name
    for suffix in (".routing", ".router"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    matches = re.findall(r"(?:^|\.)model\.(\d+(?:\.\d+)*)", normalized)
    return f"model.{matches[-1]}" if matches else normalized


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("latency artifact contains no samples")
    rank = (len(ordered) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def pooled_latency(paths: list[Path], model_sha256: str) -> dict[str, float]:
    """Pool raw per-image samples across order-controlled benchmark artifacts."""
    samples: list[float] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        batch_size = int(payload["benchmark"]["batch_size"])
        matches = [result for result in payload["results"] if result.get("model_sha256") == model_sha256]
        if len(matches) != 1:
            raise ValueError(f"expected one latency result for SHA {model_sha256} in {path}, found {len(matches)}")
        samples.extend(float(value) / batch_size for value in matches[0]["warm"]["samples_batch_ms"])
    mean = statistics.fmean(samples)
    return {
        "latency_ms": mean,
        "latency_mean_ms": mean,
        "latency_p50_ms": percentile(samples, 0.50),
        "latency_p95_ms": percentile(samples, 0.95),
        "latency_p99_ms": percentile(samples, 0.99),
        "throughput_images_s": 1000.0 / mean,
    }


def expected_structure(point: dict[str, Any]) -> dict[str, int]:
    return {canonical_layer_name(layer): len(ids) for layer, ids in point["keep_expert_ids"].items()}


def observed_structure(evaluation: dict[str, Any]) -> dict[str, int]:
    return {
        canonical_layer_name(item["layer"]): int(item["num_experts"])
        for item in evaluation["expert_signature"]
    }


def convergence_epoch(results_csv: Path | None) -> str | int:
    if results_csv is None:
        return ""
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        return ""
    key = "metrics/mAP50-95(B)"
    final = float(rows[-1][key])
    target = 0.95 * final
    for row in rows:
        if float(row[key]) >= target:
            return int(float(row["epoch"]))
    return ""


def load_measurement(entry: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
    eval_path = Path(entry["eval_json"])
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    if evaluation.get("event_count", 0) <= 0 or evaluation.get("nontrivial_signal_layer_count", 0) <= 0:
        raise ValueError(f"evaluation did not observe nontrivial routing: {eval_path}")
    selected = evaluation.get("selected_images")
    observed = evaluation.get("observed_images")
    if selected is not None and observed != selected:
        raise ValueError(f"evaluation sample-count mismatch in {eval_path}: selected={selected} observed={observed}")
    expected = expected_structure(point)
    observed_signature = observed_structure(evaluation)
    if expected != observed_signature:
        raise ValueError(f"expert structure mismatch in {eval_path}: expected={expected} observed={observed_signature}")

    layer_ginis = [float(layer["soft_contribution_gini"]) for layer in evaluation["layers"].values()]
    latency_paths = [Path(path) for path in entry["latency_jsons"]]
    model_sha256 = str(evaluation["model_sha256"])
    metrics = evaluation["metrics"]
    result: dict[str, Any] = {
        "mAP50-95": metrics["mAP50-95"],
        "mAP50": metrics["mAP50"],
        "gflops": evaluation["gflops"],
        "params_m": float(evaluation["params"]) / 1e6,
        "experts_per_layer": json.dumps(observed_signature, sort_keys=True, separators=(",", ":")),
        "expert_usage_gini": statistics.fmean(layer_ginis),
        "convergence_epoch_95pct": convergence_epoch(
            Path(entry["train_results_csv"]) if entry.get("train_results_csv") else None
        ),
        "recovery_structure_status": "preserved",
        "eval_model_sha256": model_sha256,
        "eval_json": str(eval_path),
        "latency_jsons": json.dumps([str(path) for path in latency_paths], separators=(",", ":")),
    }
    result.update(pooled_latency(latency_paths, model_sha256))
    return result


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    measurements = json.loads(args.measurements.read_text(encoding="utf-8"))["measurements"]
    plan_csv = args.plan_csv or args.manifest.with_name("moe_pruning_sweep.csv")
    with plan_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    points = {f"{float(point['threshold']):.2f}": point for point in manifest["points"]}
    for row in rows:
        point = points[f"{float(row['threshold']):.2f}"]
        representative = f"{float(point['representative_threshold']):.2f}"
        recovery = row["recovery"]
        if recovery not in RECOVERIES:
            raise ValueError(f"unsupported recovery {recovery!r}")
        entry = measurements.get(representative, {}).get(recovery)
        if entry is None:
            raise ValueError(f"missing measurement for representative={representative} recovery={recovery}")
        row.update({key: str(value) for key, value in load_measurement(entry, point).items()})

    fields = list(rows[0])
    fields.extend(field for field in MEASUREMENT_FIELDS if field not in fields)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[assemble] rows={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
