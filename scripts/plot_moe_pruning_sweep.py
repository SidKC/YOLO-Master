#!/usr/bin/env python3
"""Analyze and plot an Issue #52 MoE pruning threshold sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


QUALITY_KEY = "mAP50-95"
LATENCY_KEYS = ("latency_p95_ms", "latency_ms")
RESOURCE_KEYS = ("gflops", "params_m")
SCENARIOS = ("latency-first", "resource-first", "quality-first")
STRUCTURE_FIELDS = (
    "recovery_structure_status",
    "structure_status",
    "recovery_structure_valid",
    "structure_valid",
)
TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "y",
    "ok",
    "valid",
    "match",
    "matched",
    "preserved",
    "compatible",
    "available",
}
FALSE_VALUES = {"0", "false", "no", "n", "invalid", "mismatch", "structure_mismatch", "unavailable", "missing"}


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def as_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def latency_value(row: dict[str, str]) -> float | None:
    """Prefer measured P95 latency and retain compatibility with legacy CSVs."""
    for key in LATENCY_KEYS:
        value = as_float(row, key)
        if value is not None:
            return value
    return None


def is_no_op(row: dict[str, str]) -> bool:
    return row.get("status", "").strip().lower() == "no_op" or as_bool(row.get("no_op", "")) is True


def is_analysis_point(row: dict[str, str]) -> bool:
    """Exclude aliased duplicate thresholds while accepting legacy CSVs."""
    status = row.get("status", "").strip().lower()
    return not status or status in {"unique", "no_op"}


def objective_values(row: dict[str, str], resource: str) -> tuple[float, float, float] | None:
    quality = as_float(row, QUALITY_KEY)
    latency = latency_value(row)
    resource_value = as_float(row, resource)
    if quality is None or latency is None or resource_value is None:
        return None
    return quality, latency, resource_value


def dominates(left: tuple[float, float, float], right: tuple[float, float, float]) -> bool:
    """Return whether left dominates right (quality max; latency/resource min)."""
    return (
        left[0] >= right[0]
        and left[1] <= right[1]
        and left[2] <= right[2]
        and (left[0] > right[0] or left[1] < right[1] or left[2] < right[2])
    )


def pareto_front(rows: list[dict[str, str]], resource: str = "gflops") -> list[dict[str, str]]:
    """Compute the true three-objective non-dominated front."""
    candidates = [row for row in rows if is_analysis_point(row) and objective_values(row, resource) is not None]
    front = [
        row
        for row in candidates
        if not any(
            other is not row
            and dominates(objective_values(other, resource), objective_values(row, resource))  # type: ignore[arg-type]
            for other in candidates
        )
    ]
    def numeric_or(value: float | None, fallback: float) -> float:
        return value if value is not None else fallback

    return sorted(
        front,
        key=lambda row: (
            numeric_or(latency_value(row), float("inf")),
            numeric_or(as_float(row, resource), float("inf")),
            -numeric_or(as_float(row, QUALITY_KEY), float("-inf")),
        ),
    )


def baseline_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if is_no_op(row) and is_analysis_point(row)]


def baseline_for(row: dict[str, str], baselines: list[dict[str, str]]) -> dict[str, str] | None:
    """Prefer an anchor with matching recovery, then the strongest measured anchor."""
    recovery = row.get("recovery", "")
    matching = [candidate for candidate in baselines if candidate.get("recovery", "") == recovery]
    usable = [candidate for candidate in (matching or baselines) if as_float(candidate, QUALITY_KEY) is not None]
    return max(usable, key=lambda candidate: as_float(candidate, QUALITY_KEY) or float("-inf"), default=None)


def recovery_structure_available(row: dict[str, str], known_fields: set[str]) -> bool:
    """Require an affirmative recovery-structure result only when such a column exists."""
    if row.get("recovery", "").strip().lower() in {"", "direct", "none"}:
        return True
    fields = [field for field in STRUCTURE_FIELDS if field in known_fields]
    if not fields:
        return True
    values = [as_bool(row.get(field, "")) for field in fields]
    return any(value is True for value in values) and not any(value is False for value in values)


def feasible_rows(
    rows: list[dict[str, str]], resource: str = "gflops", max_map_drop: float = 0.01
) -> list[dict[str, str]]:
    """Filter structurally valid unique points within the baseline quality budget."""
    known_fields = {key for row in rows for key in row}
    baselines = baseline_rows(rows)
    feasible: list[dict[str, str]] = []
    for row in rows:
        if row.get("status", "").strip().lower() != "unique" or is_no_op(row):
            continue
        if objective_values(row, resource) is None or not recovery_structure_available(row, known_fields):
            continue
        baseline = baseline_for(row, baselines)
        quality = as_float(row, QUALITY_KEY)
        baseline_quality = as_float(baseline, QUALITY_KEY) if baseline is not None else None
        if quality is None or baseline_quality is None or baseline_quality - quality > max_map_drop:
            continue
        feasible.append(row)
    return feasible


def select_sweet_spots(
    rows: list[dict[str, str]], resource: str = "gflops", max_map_drop: float = 0.01
) -> dict[str, dict[str, str] | None]:
    """Select scenario-specific points after all structural and quality guards."""
    feasible = feasible_rows(rows, resource, max_map_drop)
    if not feasible:
        return {scenario: None for scenario in SCENARIOS}

    def quality(row: dict[str, str]) -> float:
        return as_float(row, QUALITY_KEY) or float("-inf")

    def latency(row: dict[str, str]) -> float:
        return latency_value(row) or float("inf")

    def resource_value(row: dict[str, str]) -> float:
        return as_float(row, resource) or float("inf")

    return {
        "latency-first": min(feasible, key=lambda row: (latency(row), resource_value(row), -quality(row))),
        "resource-first": min(feasible, key=lambda row: (resource_value(row), latency(row), -quality(row))),
        "quality-first": min(feasible, key=lambda row: (-quality(row), latency(row), resource_value(row))),
    }


def row_label(row: dict[str, str]) -> str:
    status = row.get("status", "")
    if is_no_op(row):
        return f"baseline/{row.get('recovery', '')}"
    return f"{row.get('threshold', '?')}/{row.get('recovery', '?')}" + (f"/{status}" if status else "")


def group_plot_rows(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """Group logical thresholds that share one physical structure and recovery path."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for index, row in enumerate(rows):
        recovery = row.get("recovery", "unspecified")
        signature = row.get("signature", "").strip()
        identity = signature or f"logical-row-{index}"
        grouped.setdefault((recovery, identity), []).append(row)

    def sort_key(group: list[dict[str, str]]) -> tuple[float, str]:
        thresholds = [value for row in group if (value := as_float(row, "threshold")) is not None]
        return (min(thresholds, default=float("inf")), group[0].get("recovery", ""))

    return sorted(grouped.values(), key=sort_key)


def plot_group_label(group: list[dict[str, str]]) -> str:
    """Build one compact label for an overlapping logical-threshold group."""
    thresholds = sorted({value for row in group if (value := as_float(row, "threshold")) is not None})
    if not thresholds:
        threshold_label = "?"
    elif len(thresholds) == 1:
        threshold_label = f"{thresholds[0]:.2f}"
    else:
        threshold_label = f"{thresholds[0]:.2f}-{thresholds[-1]:.2f}"
    representative = next((row for row in group if is_analysis_point(row)), group[0])
    status = "no-op" if any(is_no_op(row) for row in group) else representative.get("status", "")
    suffix = f"/{status}" if status else ""
    return f"{threshold_label}/{representative.get('recovery', '?')}{suffix}"


def analysis_record(row: dict[str, str], resource: str) -> dict[str, Any]:
    record: dict[str, Any] = dict(row)
    record["analysis_latency_ms"] = latency_value(row)
    record["analysis_latency_source"] = next((key for key in LATENCY_KEYS if as_float(row, key) is not None), None)
    record["analysis_resource"] = resource
    record["analysis_resource_value"] = as_float(row, resource)
    record["analysis_is_baseline"] = is_no_op(row)
    return record


def write_pareto_csv(path: Path, rows: list[dict[str, str]], resource: str) -> None:
    records = [analysis_record(row, resource) for row in rows]
    original_fields = list(dict.fromkeys(key for row in rows for key in row))
    analysis_fields = [
        "analysis_latency_ms",
        "analysis_latency_source",
        "analysis_resource",
        "analysis_resource_value",
        "analysis_is_baseline",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*original_fields, *analysis_fields])
        writer.writeheader()
        writer.writerows(records)


def plot_threshold_curves(rows: list[dict[str, str]], out_dir: Path, plt: Any) -> None:
    metrics: tuple[tuple[str, Any], ...] = (
        (QUALITY_KEY, lambda row: as_float(row, QUALITY_KEY)),
        ("gflops", lambda row: as_float(row, "gflops")),
        ("latency_p95_ms", latency_value),
    )
    for metric, getter in metrics:
        usable = [row for row in rows if as_float(row, "threshold") is not None and getter(row) is not None]
        if not usable:
            print(f"[plot] skip {metric}: no numeric values")
            continue
        plt.figure(figsize=(7, 4))
        recoveries = sorted({row.get("recovery", "unspecified") for row in usable})
        for recovery in recoveries:
            group = [row for row in usable if row.get("recovery", "unspecified") == recovery]
            group.sort(key=lambda row: as_float(row, "threshold") or 0.0)
            plt.plot(
                [as_float(row, "threshold") for row in group],
                [getter(row) for row in group],
                marker="o",
                label=recovery,
            )
        plt.xlabel("Pruning threshold")
        plt.ylabel("Latency P95 (ms)" if metric == "latency_p95_ms" else metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        output = out_dir / f"threshold_{metric}.png"
        plt.savefig(output, dpi=160)
        plt.close()
        print(f"[plot] wrote {output}")


def plot_pareto(
    rows: list[dict[str, str]],
    front: list[dict[str, str]],
    sweet_spots: dict[str, dict[str, str] | None],
    selected_scenarios: Iterable[str],
    out_dir: Path,
    plt: Any,
) -> None:
    usable = [row for row in rows if as_float(row, QUALITY_KEY) is not None and latency_value(row) is not None]
    if not usable:
        print("[plot] skip Pareto: need numeric P95/legacy latency and mAP50-95")
        return
    plt.figure(figsize=(7, 5))
    for group in group_plot_rows(usable):
        row = next((candidate for candidate in group if is_analysis_point(candidate)), group[0])
        marker = "s" if is_no_op(row) else "o"
        plt.scatter(latency_value(row), as_float(row, QUALITY_KEY), alpha=0.65, marker=marker)
        plt.annotate(
            plot_group_label(group),
            (latency_value(row), as_float(row, QUALITY_KEY)),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ordered = sorted(front, key=lambda row: latency_value(row) or float("inf"))
    if ordered:
        plt.plot(
            [latency_value(row) for row in ordered],
            [as_float(row, QUALITY_KEY) for row in ordered],
            linewidth=2,
            label="3-objective Pareto members",
        )
    markers = {"latency-first": "*", "resource-first": "X", "quality-first": "P"}
    for scenario in selected_scenarios:
        sweet = sweet_spots[scenario]
        if sweet is not None:
            plt.scatter(
                [latency_value(sweet)],
                [as_float(sweet, QUALITY_KEY)],
                s=130,
                marker=markers[scenario],
                label=scenario,
            )
    plt.xlabel("Latency P95 (ms; legacy fallback when absent)")
    plt.ylabel(QUALITY_KEY)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output = out_dir / "pareto_accuracy_latency.png"
    plt.savefig(output, dpi=160)
    plt.close()
    print(f"[plot] wrote {output}")


def plot_3d(rows: list[dict[str, str]], resource: str, out_dir: Path, plt: Any) -> None:
    usable = [row for row in rows if is_analysis_point(row) and objective_values(row, resource) is not None]
    if not usable:
        print(f"[plot] skip 3D: need numeric mAP50-95, latency, and {resource}")
        return
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    for row in usable:
        quality, latency, resource_value = objective_values(row, resource)  # type: ignore[misc]
        marker = "s" if is_no_op(row) else "o"
        axis.scatter(latency, resource_value, quality, marker=marker, alpha=0.75)
        axis.text(latency, resource_value, quality, row_label(row), fontsize=7)
    axis.set_xlabel("Latency P95 (ms)")
    axis.set_ylabel(resource)
    axis.set_zlabel(QUALITY_KEY)
    figure.tight_layout()
    output = out_dir / f"quality_latency_{resource}_3d.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"[plot] wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV generated by scripts/moe_pruning_sweep.py and filled with metrics.")
    parser.add_argument("--out-dir", type=Path, help="Output directory for plots. Defaults to CSV parent.")
    parser.add_argument("--resource", choices=RESOURCE_KEYS, default="gflops", help="Resource objective and 3D axis.")
    parser.add_argument(
        "--max-map-drop",
        type=float,
        default=0.01,
        help="Maximum absolute mAP50-95 drop from the matching no-op baseline (default: 0.01).",
    )
    parser.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "all"),
        default="all",
        help="Sweet Spot scenario(s) highlighted in the 2D plot (default: all).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_map_drop < 0:
        raise SystemExit("--max-map-drop must be non-negative")

    rows = read_rows(args.csv)
    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    front = pareto_front(rows, args.resource)
    sweet_spots = select_sweet_spots(rows, args.resource, args.max_map_drop)
    selected_scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)

    pareto_csv = out_dir / "pareto_front.csv"
    write_pareto_csv(pareto_csv, front, args.resource)
    report = {
        "input_csv": str(args.csv),
        "objectives": {
            "maximize": QUALITY_KEY,
            "minimize": ["latency_p95_ms (latency_ms fallback)", args.resource],
        },
        "max_map_drop": args.max_map_drop,
        "baseline_anchors": [analysis_record(row, args.resource) for row in baseline_rows(rows)],
        "pareto_front": [analysis_record(row, args.resource) for row in front],
        "sweet_spots": {
            scenario: analysis_record(row, args.resource) if row is not None else None
            for scenario, row in sweet_spots.items()
        },
    }
    report_path = out_dir / "pareto_and_sweet_spots.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[plot] wrote {pareto_csv}")
    print(f"[plot] wrote {report_path}")
    for scenario in SCENARIOS:
        sweet = sweet_spots[scenario]
        if sweet is None:
            print(f"[plot] {scenario}: no feasible point")
        else:
            print(f"[plot] {scenario}: {row_label(sweet)}")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib unavailable; analysis outputs remain available: {exc}")
        return 1

    plot_threshold_curves(rows, out_dir, plt)
    plot_pareto(rows, front, sweet_spots, selected_scenarios, out_dir, plt)
    plot_3d(rows, args.resource, out_dir, plt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
