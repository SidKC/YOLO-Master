#!/usr/bin/env python3
"""Create an auditable MoE pruning threshold sweep plan.

The plan is derived from a previously captured routing-signal JSON. All logical
thresholds remain in the manifest, while thresholds that produce the same
per-layer expert signature reuse one physical pruning surgery. Training and
evaluation are deliberately emitted as separate commands.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30)
METRIC_FIELDS = (
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
)


def fmt_threshold(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a manifest input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prune_command(
    model: Path,
    calibration_dataset: str,
    eval_dataset: str,
    signal_json: Path,
    output_model: Path,
    threshold: float,
    device: str,
    importance_mode: str,
    moe_inference_mode: str,
) -> list[str]:
    """Build the physical surgery command for one unique pruning signature."""
    return [
        sys.executable,
        "-m",
        "ultralytics.nn.modules.moe.pruning",
        str(model),
        "--output",
        str(output_model),
        "--threshold",
        f"{threshold:.2f}",
        "--dataset",
        calibration_dataset,
        "--eval-dataset",
        eval_dataset,
        "--device",
        device,
        "--importance-mode",
        importance_mode,
        "--signal-json",
        str(signal_json),
        "--moe-inference-mode",
        moe_inference_mode,
    ]


def build_val_command(model: Path, dataset: str, device: str, batch: int, imgsz: int) -> list[str]:
    return [
        str(Path(sys.executable).with_name("yolo")),
        "val",
        f"model={model}",
        f"data={dataset}",
        f"device={device}",
        f"batch={batch}",
        f"imgsz={imgsz}",
    ]


def build_lora_command(
    model: Path,
    dataset: str,
    out_dir: Path,
    device: str,
    batch: int,
    imgsz: int,
    seed: int,
) -> list[str]:
    """Build LoRA recovery training without relying on a nonexistent enable flag."""
    return [
        str(Path(sys.executable).with_name("yolo")),
        "train",
        f"model={model}",
        f"data={dataset}",
        "epochs=10",
        f"project={out_dir}",
        "name=lora_recovery",
        f"device={device}",
        f"batch={batch}",
        f"imgsz={imgsz}",
        "lora_r=16",
        "lora_alpha=32",
        "lora_backend=auto",
        "lora_include_moe=False",
        "preserve_checkpoint_structure=True",
        "amp=False",
        "deterministic=True",
        "workers=8",
        "plots=False",
        f"seed={seed}",
    ]


def shell_join(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def compare_expert_signatures(
    reference: list[tuple[str, int, int]],
    candidate: list[tuple[str, int, int]],
) -> tuple[str, str]:
    """Compare ``(layer, num_experts, top_k)`` signatures for recovery guards."""
    if reference == candidate:
        return "preserved", ""

    ref_counts = "/".join(f"{experts}:{top_k}" for _, experts, top_k in reference)
    cand_counts = "/".join(f"{experts}:{top_k}" for _, experts, top_k in candidate)
    return "structure_mismatch", f"reference={ref_counts};candidate={cand_counts}"


def load_signal(path: Path) -> dict[str, Any]:
    """Load and minimally validate the diagnosis artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = payload.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"signal JSON has no non-empty 'layers' object: {path}")
    for layer_name, layer in layers.items():
        if not isinstance(layer, dict) or not isinstance(layer.get("experts"), list) or not layer["experts"]:
            raise ValueError(f"signal JSON layer {layer_name!r} has no experts")
    required_positive = ("opportunity_count", "event_count", "nontrivial_signal_layer_count")
    invalid_counts = {
        field: payload.get(field)
        for field in required_positive
        if not isinstance(payload.get(field), (int, float)) or payload[field] <= 0
    }
    if invalid_counts:
        raise ValueError(f"signal JSON failed positive-event gates: {invalid_counts}")
    return payload


def expert_score(expert: dict[str, Any], importance_mode: str) -> float:
    """Read the selected signal using diagnosis and pruner-compatible names."""
    aliases = {
        "usage": ("hard_usage", "usage_pct"),
        "usage_weight": ("soft_contribution",),
        "avg_weight": ("average_gate_weight", "avg_weight"),
        "soft_contribution": ("soft_contribution",),
    }
    for field in aliases[importance_mode]:
        if field in expert:
            return float(expert[field])
    raise ValueError(f"expert {expert.get('expert_id', '?')} lacks signal for {importance_mode!r}")


def threshold_plan(
    layers: dict[str, dict[str, Any]], threshold: float, importance_mode: str
) -> tuple[dict[str, list[int]], bool, str]:
    """Return per-layer keep IDs, whether the plan is a no-op, and its signature."""
    keep_by_layer: dict[str, list[int]] = {}
    no_op = True
    for layer_name in sorted(layers):
        experts = layers[layer_name]["experts"]
        scored = [(int(expert["expert_id"]), expert_score(expert, importance_mode)) for expert in experts]
        keep = sorted(expert_id for expert_id, score in scored if score >= threshold)
        if not keep:
            keep = [max(scored, key=lambda item: (item[1], -item[0]))[0]]
        keep_by_layer[layer_name] = keep
        no_op = no_op and len(keep) == len(scored)
    signature = ";".join(f"{layer}:{','.join(map(str, keep_by_layer[layer]))}" for layer in sorted(keep_by_layer))
    return keep_by_layer, no_op, signature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="Trained YOLO-Master-EsMoE-N checkpoint.")
    parser.add_argument("--signal-json", type=Path, required=True, help="Output from diagnose_moe_pruning_signal.py.")
    parser.add_argument(
        "--calibration-dataset",
        "--dataset",
        dest="calibration_dataset",
        default=None,
        help="Routing-calibration dataset YAML; --dataset remains a compatibility alias.",
    )
    parser.add_argument("--eval-dataset", default=None, help="Held-out evaluation dataset YAML.")
    parser.add_argument("--train-dataset", default=None, help="Full training dataset YAML used for LoRA recovery.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/moe_pruning_sweep")
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument(
        "--importance-mode",
        choices=("usage", "usage_weight", "avg_weight", "soft_contribution"),
        default="soft_contribution",
    )
    parser.add_argument("--moe-inference-mode", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--device", default="0", help="Physical validation device, e.g. 1 or cuda:1.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute-prune", action="store_true", help="Run each unique, non-no-op surgery once.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(not 0.0 <= threshold <= 1.0 for threshold in args.thresholds):
        raise ValueError("all thresholds must be between 0.0 and 1.0")

    signal_path = args.signal_json.resolve()
    signal = load_signal(signal_path)
    model_path = args.model.resolve()
    actual_model_sha256 = sha256_file(model_path)
    expected_model_sha256 = signal.get("model_sha256")
    if expected_model_sha256 != actual_model_sha256:
        raise ValueError(
            f"signal checkpoint SHA-256 mismatch: expected={expected_model_sha256} actual={actual_model_sha256}"
        )
    calibration_dataset = args.calibration_dataset or signal.get("data")
    if not calibration_dataset:
        raise ValueError("--calibration-dataset is required when signal JSON has no 'data' field")
    eval_dataset = args.eval_dataset or calibration_dataset
    train_dataset = args.train_dataset or eval_dataset
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration_provenance = {
        "dataset": str(calibration_dataset),
        "signal_json": str(signal_path),
        "signal_sha256": sha256_file(signal_path),
        "signal_generated_at_utc": signal.get("generated_at_utc"),
        "signal_revision": signal.get("revision"),
        "signal_model": signal.get("model"),
        "signal_model_sha256": signal.get("model_sha256"),
        "signal_split": signal.get("split"),
        "signal_requested_fraction": signal.get("requested_fraction", signal.get("fraction")),
        "signal_effective_fraction": signal.get("effective_fraction"),
        "signal_available_images": signal.get("available_images"),
        "signal_selected_images": signal.get("selected_images"),
        "signal_observed_images": signal.get("observed_images"),
        "signal_observed_batches": signal.get("observed_batches"),
        "signal_batch": signal.get("batch"),
        "signal_imgsz": signal.get("imgsz"),
    }
    eval_provenance = {"dataset": str(eval_dataset), "split": "val", "batch": args.batch, "imgsz": args.imgsz}
    rows: list[dict[str, str]] = []
    manifest: dict[str, Any] = {
        "model": str(model_path),
        "model_sha256": actual_model_sha256,
        "calibration": calibration_provenance,
        "evaluation": eval_provenance,
        "training": {
            "dataset": str(train_dataset),
            "epochs": 10,
            "recovery_policy": "lora_adapters_plus_detection_head",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_include_moe": False,
            "detection_head_trainable": True,
            "amp": False,
            "deterministic": True,
            "workers": 8,
            "plots": False,
            "note": "Repository-native adapter training unfreezes the detection head.",
        },
        "importance_mode": args.importance_mode,
        "moe_inference_mode": args.moe_inference_mode,
        "device": args.device,
        "seed": args.seed,
        "thresholds": args.thresholds,
        "points": [],
        "physical_surgeries": [],
    }

    representatives: dict[str, dict[str, Any]] = {}
    for threshold in args.thresholds:
        keep_by_layer, no_op, signature = threshold_plan(signal["layers"], threshold, args.importance_mode)
        representative = representatives.get(signature)
        tag = fmt_threshold(threshold)
        point_dir = out_dir / f"threshold_{tag}"
        point_dir.mkdir(parents=True, exist_ok=True)

        if representative is None:
            status = "no_op" if no_op else "unique"
            representative_threshold = threshold
            representative_tag = tag
            representative_dir = point_dir
            pruned_model = args.model if no_op else representative_dir / f"pruned_{representative_tag}.pt"
            representatives[signature] = {
                "threshold": threshold,
                "tag": tag,
                "model": pruned_model,
                "status": status,
            }
        else:
            status = "duplicate"
            representative_threshold = float(representative["threshold"])
            representative_tag = str(representative["tag"])
            representative_dir = out_dir / f"threshold_{representative_tag}"
            pruned_model = Path(representative["model"])

        prune_cmd: list[str] = []
        if status == "unique":
            prune_cmd = build_prune_command(
                args.model,
                str(calibration_dataset),
                str(eval_dataset),
                signal_path,
                pruned_model,
                representative_threshold,
                args.device,
                args.importance_mode,
                args.moe_inference_mode,
            )
            manifest["physical_surgeries"].append(
                {
                    "representative_threshold": representative_threshold,
                    "signature": signature,
                    "output_model": str(pruned_model),
                    "command": prune_cmd,
                }
            )
            if args.execute_prune:
                subprocess.run(prune_cmd, check=True, cwd=ROOT)

        direct_eval_cmd = build_val_command(pruned_model, str(eval_dataset), args.device, args.batch, args.imgsz)
        lora_cmd = build_lora_command(
            pruned_model, str(train_dataset), representative_dir, args.device, args.batch, args.imgsz, args.seed
        )
        recovered_model = representative_dir / "lora_recovery/weights/best.pt"
        lora_eval_cmd = build_val_command(recovered_model, str(eval_dataset), args.device, args.batch, args.imgsz)

        common = {
            "threshold": f"{threshold:.2f}",
            "status": status,
            "representative_threshold": f"{representative_threshold:.2f}",
            "reused_from": f"{representative_threshold:.2f}" if status == "duplicate" else "",
            "signature": signature,
            "calibration_dataset": str(calibration_dataset),
            "eval_dataset": str(eval_dataset),
            "train_dataset": str(train_dataset),
            "signal_json": str(signal_path),
            "model": str(pruned_model),
            "prune_command": shell_join(prune_cmd),
        }
        for recovery, recovery_command, eval_command in (
            ("direct", [], direct_eval_cmd),
            ("lora10", lora_cmd, lora_eval_cmd),
        ):
            row = {
                **common,
                "recovery": recovery,
                "recovery_command": shell_join(recovery_command),
                "eval_command": shell_join(eval_command),
            }
            row.update({field: "" for field in METRIC_FIELDS})
            rows.append(row)

        manifest["points"].append(
            {
                "threshold": threshold,
                "status": status,
                "no_op": no_op,
                "unique_signature": status != "duplicate",
                "signature": signature,
                "keep_expert_ids": keep_by_layer,
                "representative_threshold": representative_threshold,
                "reused_from": representative_threshold if status == "duplicate" else None,
                "model": str(pruned_model),
                "prune_command": prune_cmd,
                "direct": {"eval_command": direct_eval_cmd},
                "lora10": {
                    "recovery_command": lora_cmd,
                    "eval_model": str(recovered_model),
                    "eval_command": lora_eval_cmd,
                },
            }
        )

    csv_path = out_dir / "moe_pruning_sweep.csv"
    fieldnames = [
        "threshold",
        "status",
        "representative_threshold",
        "reused_from",
        "signature",
        "recovery",
        "calibration_dataset",
        "eval_dataset",
        "train_dataset",
        "signal_json",
        "model",
        "prune_command",
        "recovery_command",
        "eval_command",
        *METRIC_FIELDS,
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = out_dir / "moe_pruning_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[moe-sweep] logical_points={len(args.thresholds)} physical_surgeries={len(manifest['physical_surgeries'])}")
    print(f"[moe-sweep] wrote {csv_path}")
    print(f"[moe-sweep] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
