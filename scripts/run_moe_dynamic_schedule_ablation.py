#!/usr/bin/env python3
"""Run issue #52 dynamic MoE schedule comparison on VisDrone.

The script compares three groups:

- `baseline`: fixed MoE balance coefficient.
- `dynamic`: Gini-driven dynamic balance coefficient.
- `ablation`: fixed low balance coefficient.

Example:

    python scripts/run_moe_dynamic_schedule_ablation.py --dry-run
    python scripts/run_moe_dynamic_schedule_ablation.py --variant dynamic --epochs 100 --wandb offline
    python scripts/run_moe_dynamic_schedule_ablation.py --summary-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/reproduce/_runtime/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/reproduce/_runtime/matplotlib"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import SETTINGS, YAML  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402


METRIC_KEY = "metrics/mAP50-95(B)"
MAP50_KEY = "metrics/mAP50(B)"
DYNAMIC_GINI_KEY = "moe/dynamic_gini"
DYNAMIC_BALANCE_KEY = "moe/dynamic_balance_loss_coeff"
COUNT_KEYS = {
    "final_opportunity_count": "moe/dynamic_opportunity_count",
    "final_event_count": "moe/dynamic_event_count",
    "final_nontrivial_action_count": "moe/dynamic_nontrivial_action_count",
}
OBSERVATION_KEYS = {
    "final_routing_observation_count": "moe/dynamic_routing_observation_count",
    "final_min_layer_observation_count": "moe/dynamic_min_layer_observation_count",
    "final_max_layer_observation_count": "moe/dynamic_max_layer_observation_count",
}


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    scheduler_impl: str
    formula_version: str
    extra_args: dict[str, Any]


VARIANTS = {
    "baseline": Variant(
        key="baseline",
        name="visdrone_issue52_fixed_balance",
        scheduler_impl="fixed_balance",
        formula_version="fixed_balance_v1",
        extra_args={"moe_dynamic_schedule": "none", "moe_balance_loss": 1.0},
    ),
    "dynamic": Variant(
        key="dynamic",
        name="visdrone_issue52_gini_balance",
        scheduler_impl="GiniBalanceScheduler",
        formula_version="gini_ema_exp_clip_v1",
        extra_args={
            "moe_dynamic_schedule": "gini_balance",
            "moe_balance_loss": 1.0,
            "moe_dynamic_gini_target": 0.25,
            "moe_dynamic_gini_alpha": 1.0,
            "moe_dynamic_gini_beta": 0.8,
            "moe_dynamic_balance_min": 0.5,
            "moe_dynamic_balance_max": 2.0,
        },
    ),
    "ablation": Variant(
        key="ablation",
        name="visdrone_issue52_low_balance",
        scheduler_impl="fixed_low_balance",
        formula_version="fixed_balance_v1",
        extra_args={"moe_dynamic_schedule": "none", "moe_balance_loss": 0.3},
    ),
}


def default_model_cfg() -> Path:
    return ROOT / "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml"


def default_data_yaml() -> Path:
    local = ROOT / "runs/reproduce/visdrone/_data/VisDrone.local.yaml"
    return local if local.exists() else ROOT / "ultralytics/cfg/datasets/VisDrone.yaml"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def configure_wandb(mode: str) -> None:
    if mode == "disabled":
        SETTINGS.update({"wandb": False})
        os.environ.setdefault("WANDB_DISABLED", "true")
        return
    SETTINGS.update({"wandb": True})
    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_MODE"] = "offline" if mode == "offline" else "online"


def read_results(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.exists():
        return []
    with results_csv.open(newline="", encoding="utf-8") as handle:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON objects from a JSONL artifact."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def first_epoch_at(rows: list[dict[str, str]], target: float) -> int | float | None:
    """Return the explicit results.csv epoch value at which the target is first reached."""
    for row in rows:
        if as_float(row, METRIC_KEY) >= target:
            epoch = as_float(row, "epoch")
            if math.isfinite(epoch):
                return int(epoch) if epoch.is_integer() else epoch
    return None


def finite_values(rows: list[dict[str, str]], key: str) -> list[float]:
    """Collect finite values for an optional results.csv metric."""
    values = [as_float(row, key) for row in rows]
    return [value for value in values if math.isfinite(value)]


def run_args(run_dir: Path) -> dict[str, Any]:
    """Load the resolved trainer arguments saved beside a run, if available."""
    args_yaml = run_dir / "args.yaml"
    return YAML.load(args_yaml) if args_yaml.exists() else {}


def audit_dynamic_trace(project: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    """Verify that each accepted dynamic epoch has exactly one valid trace record."""
    trace_path = project / VARIANTS["dynamic"].name / "moe_dynamic_trace.jsonl"
    records = read_jsonl(trace_path)
    expected_epochs = [int(as_float(row, "epoch")) for row in rows if math.isfinite(as_float(row, "epoch"))]
    recorded_epochs = [int(record.get("epoch", -1)) for record in records]
    errors = []
    if not expected_epochs:
        errors.append("no accepted dynamic epochs were found")
    if not records:
        errors.append("dynamic trace is missing or empty")
    if recorded_epochs != expected_epochs:
        errors.append(f"trace epochs {recorded_epochs} do not match accepted epochs {expected_epochs}")
    if any(int(record.get("layer_event_count", 0)) <= 0 for record in records):
        errors.append("one or more trace records have no layer events")
    if any(int(record.get("routing_observation_count", 0)) <= 0 for record in records):
        errors.append("one or more trace records have no routing observations")
    if any(int(record.get("min_layer_observation_count", 0)) <= 0 for record in records):
        errors.append("one or more trace records have an unobserved layer")

    opportunities = [int(record.get("opportunity_count", 0)) for record in records]
    events = [int(record.get("event_count", 0)) for record in records]
    actions = [int(record.get("nontrivial_action_count", 0)) for record in records]
    expected_opportunities = list(range(1, len(records) + 1))
    if opportunities != expected_opportunities:
        errors.append(f"opportunity counts {opportunities} do not equal {expected_opportunities}")
    if events and (events != sorted(events) or events[-1] <= 0):
        errors.append("event counts must be monotonic and end above zero")
    if actions and (actions != sorted(actions) or actions[-1] <= 0):
        errors.append("nontrivial-action counts must be monotonic and end above zero")

    audit = {
        "valid": not errors,
        "trace_path": rel(trace_path),
        "accepted_epoch_count": len(expected_epochs),
        "trace_record_count": len(records),
        "expected_epochs": expected_epochs,
        "recorded_epochs": recorded_epochs,
        "final_opportunity_count": opportunities[-1] if opportunities else 0,
        "final_event_count": events[-1] if events else 0,
        "final_nontrivial_action_count": actions[-1] if actions else 0,
        "errors": errors,
    }
    audit_path = project / "dynamic_schedule_trace_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def model_state_sha256(model: Any) -> str:
    """Hash a model state deterministically for cross-variant initialization audits."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def record_initialization(project: Path, variant: Variant, model_hash: str, args: argparse.Namespace) -> Path:
    """Atomically write one variant record without a shared read-modify-write race."""
    path = project / f"dynamic_schedule_initialization.{variant.key}.json"
    payload = {
        "variant": variant.key,
        "run_name": variant.name,
        "model_state_sha256": model_hash,
        "model_source": str(args.model),
        "data": str(args.data),
        "seed": args.seed,
        "deterministic": args.deterministic,
    }
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if existing and existing.get("model_state_sha256") != model_hash:
        raise RuntimeError(
            f"initialization hash changed for {variant.key}: "
            f"{existing.get('model_state_sha256')} != {model_hash}"
        )
    peer_hashes = {}
    for peer_path in project.glob("dynamic_schedule_initialization.*.json"):
        peer = json.loads(peer_path.read_text(encoding="utf-8"))
        if peer.get("variant") != variant.key:
            peer_hashes[str(peer.get("variant", peer_path.stem))] = peer.get("model_state_sha256")
    legacy_path = project / "dynamic_schedule_initialization.json"
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        peer_hashes.update(
            {
                key: value.get("model_state_sha256")
                for key, value in legacy.get("variants", {}).items()
                if key != variant.key
            }
        )
    mismatched_peers = {key: value for key, value in peer_hashes.items() if value != model_hash}
    if mismatched_peers:
        raise RuntimeError(f"cross-variant initialization mismatch for {variant.key}: {mismatched_peers}")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def audit_initializations(project: Path, variants: list[Variant]) -> dict[str, Any]:
    """Merge per-variant or legacy records and fail closed on mismatched starts."""
    legacy_path = project / "dynamic_schedule_initialization.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8")) if legacy_path.exists() else {}
    records = {}
    errors = []
    common_fields = ("model_source", "data", "seed", "deterministic")
    common_values: dict[str, Any] = {}
    for variant in variants:
        path = project / f"dynamic_schedule_initialization.{variant.key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
        elif variant.key in legacy.get("variants", {}):
            record = {
                **{field: legacy.get(field) for field in common_fields},
                "variant": variant.key,
                **legacy["variants"][variant.key],
            }
        else:
            errors.append(f"missing initialization record for {variant.key}")
            continue
        records[variant.key] = {
            "run_name": record.get("run_name", variant.name),
            "model_state_sha256": record.get("model_state_sha256"),
        }
        for field in common_fields:
            value = record.get(field)
            if field in common_values and common_values[field] != value:
                errors.append(f"initialization field {field} differs for {variant.key}")
            else:
                common_values[field] = value

    hashes = {record.get("model_state_sha256") for record in records.values()}
    if None in hashes or len(hashes) != 1:
        errors.append(f"initial model hashes differ: {sorted(str(value) for value in hashes)}")
    payload = {**common_values, "valid": not errors, "errors": errors, "variants": records}
    temporary = legacy_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(legacy_path)
    return payload


def summarize(project: Path, variants: list[Variant], strict_trace: bool = True) -> Path:
    rows_by_variant = {variant.key: read_results(project / variant.name / "results.csv") for variant in variants}
    initialization_audit = audit_initializations(project, variants)
    trace_audit = audit_dynamic_trace(project, rows_by_variant.get("dynamic", []))
    baseline_rows = rows_by_variant.get("baseline", [])
    baseline_final = as_float(baseline_rows[-1], METRIC_KEY) if baseline_rows else float("nan")
    target = baseline_final * 0.95 if baseline_final == baseline_final else float("nan")
    baseline_epoch = first_epoch_at(baseline_rows, target) if target == target else None

    summary_rows = []
    for variant in variants:
        rows = rows_by_variant[variant.key]
        final = rows[-1] if rows else {}
        best = max(rows, key=lambda row: as_float(row, METRIC_KEY)) if rows else {}
        reach_epoch = first_epoch_at(rows, target) if target == target else None
        resolved_args = run_args(project / variant.name)
        gini_values = finite_values(rows, DYNAMIC_GINI_KEY)
        balance_values = finite_values(rows, DYNAMIC_BALANCE_KEY)
        optional_metrics = {
            **{field: final.get(key, "") for field, key in COUNT_KEYS.items()},
            **{field: final.get(key, "") for field, key in OBSERVATION_KEYS.items()},
            "final_gini": final.get(DYNAMIC_GINI_KEY, ""),
            "mean_gini": (sum(gini_values) / len(gini_values)) if gini_values else "",
            "min_gini": min(gini_values) if gini_values else "",
            "max_gini": max(gini_values) if gini_values else "",
            "final_balance_loss_coeff": final.get(DYNAMIC_BALANCE_KEY, ""),
            "mean_balance_loss_coeff": (sum(balance_values) / len(balance_values)) if balance_values else "",
            "min_balance_loss_coeff": min(balance_values) if balance_values else "",
            "max_balance_loss_coeff": max(balance_values) if balance_values else "",
        }
        summary_rows.append(
            {
                "variant": variant.key,
                "run_dir": rel(project / variant.name),
                "scheduler_impl": variant.scheduler_impl,
                "formula_version": variant.formula_version,
                "train_fraction": resolved_args.get("fraction", ""),
                "train_seed": resolved_args.get("seed", ""),
                "train_imgsz": resolved_args.get("imgsz", ""),
                "train_batch": resolved_args.get("batch", ""),
                "epochs": len(rows),
                "final_mAP50-95": final.get(METRIC_KEY, ""),
                "final_mAP50": final.get(MAP50_KEY, ""),
                "best_mAP50-95": best.get(METRIC_KEY, ""),
                "best_mAP50": best.get(MAP50_KEY, ""),
                "target_95pct_baseline_mAP50-95": target if target == target else "",
                "epoch_to_target": reach_epoch if reach_epoch is not None else "",
                "convergence_epoch_ratio": (
                    reach_epoch / baseline_epoch
                    if reach_epoch is not None and baseline_epoch is not None and baseline_epoch > 0
                    else ""
                ),
                "trace_audit_status": (
                    ("valid" if trace_audit["valid"] else "invalid")
                    if variant.key == "dynamic"
                    else "not_applicable"
                ),
                "trace_record_count": trace_audit["trace_record_count"] if variant.key == "dynamic" else "",
                "initialization_audit_status": "valid" if initialization_audit["valid"] else "invalid",
                **optional_metrics,
            }
        )

    out = project / "dynamic_schedule_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "run_dir",
        "scheduler_impl",
        "formula_version",
        "train_fraction",
        "train_seed",
        "train_imgsz",
        "train_batch",
        "epochs",
        "final_mAP50-95",
        "final_mAP50",
        "best_mAP50-95",
        "best_mAP50",
        "target_95pct_baseline_mAP50-95",
        "epoch_to_target",
        "convergence_epoch_ratio",
        "trace_audit_status",
        "trace_record_count",
        "initialization_audit_status",
        *COUNT_KEYS,
        *OBSERVATION_KEYS,
        "final_gini",
        "mean_gini",
        "min_gini",
        "max_gini",
        "final_balance_loss_coeff",
        "mean_balance_loss_coeff",
        "min_balance_loss_coeff",
        "max_balance_loss_coeff",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    if strict_trace:
        failures = []
        if not initialization_audit["valid"]:
            failures.append("initialization audit: " + "; ".join(initialization_audit["errors"]))
        if not trace_audit["valid"]:
            failures.append("dynamic trace audit: " + "; ".join(trace_audit["errors"]))
        if failures:
            raise RuntimeError("; ".join(failures))
    return out


def selected_variants(value: str) -> list[Variant]:
    if value == "all":
        return [VARIANTS["baseline"], VARIANTS["dynamic"], VARIANTS["ablation"]]
    return [VARIANTS[value]]


def require_fresh_run_dir(project: Path, variant: Variant) -> Path:
    """Refuse stale outputs because fixed run names and append-only traces cannot be safely reused."""
    run_dir = project / variant.name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}; use a new --project directory")
    return run_dir


def fraction_arg(value: str) -> float:
    """Parse a training-data fraction in the interval (0, 1]."""
    fraction = float(value)
    if not 0.0 < fraction <= 1.0:
        raise argparse.ArgumentTypeError("fraction must be in the interval (0, 1]")
    return fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("all", "baseline", "dynamic", "ablation"), default="all")
    parser.add_argument("--model", type=Path, default=default_model_cfg())
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument("--project", type=Path, default=ROOT / "runs/reproduce/issue52_dynamic_schedule")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--fraction", type=fraction_arg, default=1.0, help="fraction of the training dataset to use, in (0, 1]"
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(deterministic=True, amp=True)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--wandb", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model = resolve(args.model)
    args.data = resolve(args.data)
    args.project = resolve(args.project)
    configure_wandb(args.wandb)
    variants = selected_variants(args.variant)

    print(f"[issue52-dynamic] model={rel(args.model)}")
    print(f"[issue52-dynamic] data={rel(args.data)}")
    print(f"[issue52-dynamic] project={rel(args.project)}")
    for variant in variants:
        print(f"  - {variant.key}: {variant.name} {variant.extra_args}")

    if args.dry_run:
        return 0

    if args.summary_only:
        summary = summarize(
            args.project, [VARIANTS["baseline"], VARIANTS["dynamic"], VARIANTS["ablation"]], strict_trace=True
        )
        print(f"[summary] {rel(summary)}")
        return 0

    args.project.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        require_fresh_run_dir(args.project, variant)
        init_seeds(args.seed, deterministic=args.deterministic)
        model = YOLO(str(args.model))
        initialization_hash = model_state_sha256(model.model)
        initialization_manifest = record_initialization(args.project, variant, initialization_hash, args)
        print(
            f"[issue52-dynamic] {variant.key} initialization={initialization_hash} "
            f"manifest={initialization_manifest}"
        )
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            fraction=args.fraction,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            deterministic=args.deterministic,
            patience=args.patience,
            amp=args.amp,
            plots=args.plots,
            project=str(args.project),
            name=variant.name,
            exist_ok=args.exist_ok,
            save_period=args.save_period,
            pretrained=False,
            lora_r=0,
            **variant.extra_args,
        )

    summary = summarize(
        args.project,
        [VARIANTS["baseline"], VARIANTS["dynamic"], VARIANTS["ablation"]],
        strict_trace=args.variant in {"all", "dynamic"},
    )
    print(f"[summary] {rel(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
