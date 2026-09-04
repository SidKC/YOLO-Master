#!/usr/bin/env python3
"""Generate the public B1 P1 figures from the result CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


CONDITION_ORDER = ("random", "fixed", "true", "zero", "wrong")
CONDITION_LABELS = {
    "random": "deterministic-random",
    "fixed": "fixed-balanced",
    "true": "true-text",
    "zero": "zero-text",
    "wrong": "wrong-new",
}
SEED_COLORS = {0: "#4C72B0", 1: "#E17C05", 2: "#55A868"}
TEXT_COLOR = "#202832"
MUTED_COLOR = "#5E6875"
GRID_COLOR = "#DDE2E7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Result CSV is empty: {path}")
    required = {
        "seed",
        "label",
        "a_key",
        "b_key",
        "new17_ap_a",
        "new17_ap_b",
        "delta_ap",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Result CSV is missing columns: {sorted(missing)}")
    return rows


def condition_for_key(key: str) -> tuple[int, str]:
    parts = key.split("__", 2)
    if len(parts) != 3 or not parts[0].startswith("seed"):
        raise ValueError(f"Unexpected condition key: {key}")
    seed = int(parts[0][4:])
    route = parts[1]
    evaluation = parts[2]
    if route == "deterministic-random-routing" and evaluation == "native-random":
        return seed, "random"
    if route == "fixed-balanced-routing" and evaluation == "native-fixed":
        return seed, "fixed"
    if route == "true-text" and evaluation == "true-new17":
        return seed, "true"
    if route == "true-text" and evaluation == "zero":
        return seed, "zero"
    if route == "true-text" and evaluation == "wrong-new":
        return seed, "wrong"
    raise ValueError(f"Unexpected condition key: {key}")


def collect_condition_values(rows: list[dict[str, str]]) -> dict[int, dict[str, float]]:
    values: dict[int, dict[str, float]] = {}
    for row in rows:
        for key_field, value_field in (("a_key", "new17_ap_a"), ("b_key", "new17_ap_b")):
            seed, condition = condition_for_key(row[key_field])
            value = float(row[value_field])
            current = values.setdefault(seed, {}).get(condition)
            if current is not None and not math.isclose(current, value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Conflicting values for seed {seed}/{condition}: {current} vs {value}")
            values[seed][condition] = value
    seeds = sorted(values)
    if seeds != [0, 1, 2]:
        raise ValueError(f"Expected seeds [0, 1, 2], found {seeds}")
    for seed in seeds:
        missing = set(CONDITION_ORDER).difference(values[seed])
        if missing:
            raise ValueError(f"Seed {seed} is missing conditions: {sorted(missing)}")
    return values


def collect_deltas(rows: list[dict[str, str]], label: str, sign: float = 1.0) -> dict[int, float]:
    matches = [row for row in rows if row["label"] == label]
    if len(matches) != 3:
        raise ValueError(f"Expected three rows for {label}, found {len(matches)}")
    deltas = {int(row["seed"]): sign * float(row["delta_ap"]) * 100.0 for row in matches}
    if sorted(deltas) != [0, 1, 2]:
        raise ValueError(f"Expected seeds [0, 1, 2] for {label}, found {sorted(deltas)}")
    return deltas


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "svg.hashsalt": "b1-p1-attribution-closure",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": MUTED_COLOR,
            "axes.linewidth": 0.75,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.65,
            "grid.alpha": 0.9,
            "xtick.color": "#38424C",
            "ytick.color": "#38424C",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "text.color": TEXT_COLOR,
        }
    )


def save_svg(figure: plt.Figure, path: Path) -> Path:
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
        metadata={"Date": None},
    )
    content = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in content.splitlines()) + "\n", encoding="utf-8")
    return path


def plot_condition_ap(values: dict[int, dict[str, float]], output_dir: Path) -> Path:
    seeds = [0, 1, 2]
    rows = ("random", "true", "fixed", "wrong", "zero")
    relative_values = {
        seed: {
            condition: (values[seed][condition] - values[seed]["random"]) * 100.0
            for condition in rows
        }
        for seed in seeds
    }
    maximum_abs_value = max(abs(value) for seed in seeds for value in relative_values[seed].values())
    axis_limit = max(1.0, math.ceil(maximum_abs_value * 1.05 * 10.0) / 10.0)
    tick_count = math.floor(axis_limit / 0.5)
    tick_values = [round(index * 0.5, 2) for index in range(-tick_count, tick_count + 1)]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(10.4, 4.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    figure.get_layout_engine().set(rect=(0.0, 0.10, 1.0, 0.91))
    primary_color = "#D62728"
    context_color = "#8C8C8C"
    baseline_color = "#37474F"
    row_labels = {
        "random": "deterministic-random\n(baseline)",
        "true": "true-text",
        "fixed": "fixed-balanced",
        "wrong": "wrong-new",
        "zero": "zero-text",
    }
    for axis, seed in zip(axes, seeds, strict=True):
        axis.axvline(0.0, color=baseline_color, linewidth=1.0, zorder=1)
        for row_index, condition in enumerate(rows):
            value = relative_values[seed][condition]
            if condition == "random":
                axis.scatter(
                    value,
                    row_index,
                    s=86,
                    marker="D",
                    color=baseline_color,
                    edgecolor="white",
                    linewidth=1.0,
                    zorder=4,
                )
            elif condition == "true":
                axis.scatter(
                    value,
                    row_index,
                    s=156,
                    marker="o",
                    color=primary_color,
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=5,
                )
            else:
                axis.scatter(
                    value,
                    row_index,
                    s=70,
                    marker="o",
                    color=context_color,
                    edgecolor="white",
                    linewidth=0.9,
                    zorder=3,
                )
        axis.set_xlim(-axis_limit, axis_limit)
        axis.set_xticks(tick_values)
        axis.xaxis.set_major_formatter(FuncFormatter(format_ap_points))
        axis.set_ylim(-0.6, len(rows) - 0.4)
        axis.invert_yaxis()
        axis.set_title(f"seed {seed}", pad=10)
        axis.set_xlabel("Relative AP (points)")
        axis.grid(axis="x", visible=True)
        axis.grid(axis="y", visible=False)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(range(len(rows))), [row_labels[condition] for condition in rows])
    axes[0].set_ylabel("Routing condition")
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    for tick_label, condition in zip(axes[0].get_yticklabels(), rows, strict=True):
        if condition == "true":
            tick_label.set_color(primary_color)
            tick_label.set_fontweight("bold")
        elif condition == "random":
            tick_label.set_color(baseline_color)
    figure.suptitle("B1 P1: New17 AP relative to deterministic-random", fontsize=15, fontweight="bold")
    figure.text(
        0.5,
        0.035,
        "Right of zero = above baseline · exact AP values are reported in the table",
        ha="center",
        fontsize=10.5,
        color=MUTED_COLOR,
    )
    output = output_dir / "b1_p1_attribution_closure_ap.svg"
    save_svg(figure, output)
    plt.close(figure)
    return output


def format_ap_points(value: float, _position: float) -> str:
    if abs(value) < 1e-12:
        return "0.00"
    return f"{value:+.2f}"


def plot_deltas(rows: list[dict[str, str]], output_dir: Path) -> Path:
    comparisons = (
        (
            "true-text − deterministic-random",
            collect_deltas(rows, "native_random_minus_true_text_mixed_descriptive", sign=-1.0),
            (-0.8, 0.8),
            0.2,
        ),
        (
            "true-text − zero-text",
            collect_deltas(rows, "true_new17_minus_zero_assignment_fixed"),
            (-0.04, 0.04),
            0.01,
        ),
        (
            "true-text − wrong-new",
            collect_deltas(rows, "true_new17_minus_wrong_new_assignment_fixed"),
            (-0.02, 0.16),
            0.04,
        ),
    )
    seeds = [0, 1, 2]
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 5.8), sharey=True)
    figure.subplots_adjust(left=0.15, right=0.98, top=0.81, bottom=0.21, wspace=0.18)
    y_positions = [0, 1, 2]
    for axis, (title, delta_by_seed, x_limits, tick_step) in zip(axes, comparisons, strict=True):
        for y_position, seed in zip(y_positions, seeds, strict=True):
            value = delta_by_seed[seed]
            axis.hlines(y_position, 0.0, value, color=SEED_COLORS[seed], linewidth=3.0, alpha=0.75, zorder=1)
            axis.scatter(
                value,
                y_position,
                s=105,
                color=SEED_COLORS[seed],
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
            offset = max((x_limits[1] - x_limits[0]) * 0.025, 0.003)
            axis.text(
                value + (offset if value >= 0 else -offset),
                y_position,
                format_ap_points(value, 0.0),
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=11,
                color=TEXT_COLOR,
            )
        axis.axvline(0.0, color=MUTED_COLOR, linewidth=0.9)
        axis.set_xlim(*x_limits)
        axis.set_ylim(-0.65, 2.65)
        axis.invert_yaxis()
        axis.xaxis.set_major_locator(MultipleLocator(tick_step))
        axis.xaxis.set_major_formatter(FuncFormatter(format_ap_points))
        axis.set_title(title, loc="left", pad=12, fontsize=12)
        axis.set_xlabel("AP points")
        axis.grid(axis="y", visible=False)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(y_positions, [f"Seed {seed}" for seed in seeds])
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    axes[0].set_ylabel("Random seed")
    figure.suptitle("B1 P1: New17 AP differences for true-text", fontsize=16, fontweight="bold", y=0.97)
    figure.text(0.5, 0.055, "Positive values favor true-text · each panel uses its own AP-point scale", ha="center", fontsize=11, color=MUTED_COLOR)
    output = output_dir / "b1_p1_attribution_closure_deltas.svg"
    save_svg(figure, output)
    plt.close(figure)
    return output


def main() -> None:
    arguments = parse_args()
    rows = read_results(arguments.results_csv)
    values = collect_condition_values(rows)
    configure_style()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for output in (plot_condition_ap(values, arguments.output_dir), plot_deltas(rows, arguments.output_dir)):
        print(output)


if __name__ == "__main__":
    main()
