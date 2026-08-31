#!/usr/bin/env python3
"""Generate reproducible figures for the B1 P1-2 text-routing report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np


TRUE_COLOR = "#4C72B0"
ZERO_COLOR = "#9EA7B3"
EFFECT_COLOR = "#E17C05"
REFERENCE_COLOR = "#55A868"
TEXT_COLOR = "#202832"
MUTED_COLOR = "#5E6875"
GRID_COLOR = "#DDE2E7"
PANEL_COLOR = "#F7F9FB"
SUBSETS = (("overall65", "Overall 65"), ("base48", "Base 48"), ("new17", "New 17"))
METRICS = ("AP", "AP50", "AP75")


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
    return rows


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.edgecolor": MUTED_COLOR,
            "axes.linewidth": 0.75,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.65,
            "grid.alpha": 0.9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "svg.hashsalt": "b1-p1-2-text-causal",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "xtick.color": "#38424C",
            "ytick.color": "#38424C",
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "text.color": TEXT_COLOR,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    png_output = stem.with_suffix(".png")
    svg_output = stem.with_suffix(".svg")
    save_options = {"bbox_inches": "tight", "pad_inches": 0.08, "facecolor": "white"}
    fig.savefig(png_output, **save_options)
    fig.savefig(svg_output, metadata={"Date": None}, **save_options)
    svg_text = svg_output.read_text(encoding="utf-8")
    svg_output.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    return [png_output, svg_output]


def find_row(rows: list[dict[str, str]], section: str, subset: str, metric: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["section"] == section and row["subset"] == subset and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {section}/{subset}/{metric}, found {len(matches)}")
    return matches[0]


def plot_quality(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.55), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.70, bottom=0.19, wspace=0.12)
    y_positions = np.arange(len(METRICS))[::-1]

    for ax, (subset_key, subset_label) in zip(axes, SUBSETS, strict=True):
        subset_rows = [find_row(rows, "evaluation", subset_key, metric) for metric in METRICS]
        for y, row in zip(y_positions, subset_rows, strict=True):
            true_value = float(row["true_text"])
            zero_value = float(row["zero_text"])
            delta = float(row["delta"])
            ax.annotate(
                "",
                xy=(true_value, y),
                xytext=(zero_value, y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": EFFECT_COLOR,
                    "linewidth": 1.7,
                    "mutation_scale": 11,
                    "shrinkA": 5,
                    "shrinkB": 5,
                },
                zorder=2,
            )
            ax.scatter(
                zero_value,
                y,
                s=72,
                color=ZERO_COLOR,
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
            ax.scatter(
                true_value,
                y,
                s=82,
                color=TRUE_COLOR,
                edgecolor="white",
                linewidth=0.9,
                zorder=4,
            )
            ax.text(
                zero_value,
                y - 0.21,
                f"{zero_value:.4f}",
                ha="center",
                va="top",
                color=MUTED_COLOR,
                fontsize=8.1,
            )
            ax.text(
                true_value,
                y + 0.19,
                f"{true_value:.4f}",
                ha="center",
                va="bottom",
                color=TRUE_COLOR,
                fontsize=8.3,
                fontweight="bold",
            )
            ax.text(
                true_value + 0.008,
                y,
                f"+{delta:.4f}",
                ha="left",
                va="center",
                color=EFFECT_COLOR,
                fontsize=8.2,
                fontweight="bold",
            )

        ax.set_title(subset_label, loc="left", pad=11)
        ax.set_xlim(0.145, 0.335)
        ax.set_ylim(-0.55, 2.55)
        ax.set_xticks((0.16, 0.20, 0.24, 0.28, 0.32))
        ax.grid(axis="y", visible=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y_positions, METRICS)
    axes[0].set_ylabel("Metric", labelpad=10)
    for ax in axes:
        ax.set_xlabel("COCO box AP")

    legend_handles = (
        Line2D([], [], marker="o", linestyle="none", markersize=7, color=TRUE_COLOR, label="True text"),
        Line2D([], [], marker="o", linestyle="none", markersize=7, color=ZERO_COLOR, label="Zero text"),
        Line2D([], [], color=EFFECT_COLOR, linewidth=1.7, label="Paired gain"),
    )
    fig.legend(
        legend_handles,
        [item.get_label() for item in legend_handles],
        loc="center",
        bbox_to_anchor=(0.5, 0.805),
        ncols=3,
    )
    fig.suptitle("Paired COCO evaluation", fontsize=15, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.905,
        "Each arrow points from zero-text to true-text under matched training and evaluation settings",
        ha="center",
        fontsize=9,
        color=MUTED_COLOR,
    )
    fig.text(
        0.5,
        0.04,
        "5,000 COCO val2017 images · fixed 65-class classifier · identical evaluation settings",
        ha="center",
        fontsize=8.6,
        color=MUTED_COLOR,
    )
    outputs = save_figure(fig, output_dir / "b1_p1_2_text_causal_quality")
    plt.close(fig)
    return outputs


def plot_new17_context(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    new17 = find_row(rows, "evaluation", "new17", "AP")
    reference = find_row(rows, "reference", "released_adapter_off", "new17_AP")
    values = np.array([float(new17["zero_text"]), float(new17["true_text"]), float(reference["value"])])
    labels = ("Zero text", "True text", "Released adapter-off")
    colors = (ZERO_COLOR, TRUE_COLOR, REFERENCE_COLOR)
    y_positions = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(9.7, 4.4))
    fig.subplots_adjust(left=0.22, right=0.96, top=0.77, bottom=0.24)
    x_min = 0.14
    for y, value, color in zip(y_positions, values, colors, strict=True):
        ax.plot([x_min, value], [y, y], color=to_rgba(color, 0.22), linewidth=4.5, solid_capstyle="round")
        ax.scatter(value, y, s=105, color=color, edgecolor="white", linewidth=1.1, zorder=3)
        ax.text(
            value + 0.006,
            y,
            f"{value:.5f}",
            ha="left",
            va="center",
            fontsize=9.2,
            color=color,
            fontweight="bold",
        )

    zero_value = values[0]
    true_value = values[1]
    effect_y = 2.58
    for value, point_y in ((zero_value, y_positions[0]), (true_value, y_positions[1])):
        ax.vlines(
            value,
            point_y + 0.10,
            effect_y,
            color=to_rgba(EFFECT_COLOR, 0.65),
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            zorder=1,
        )
    effect_arrow = FancyArrowPatch(
        (zero_value, effect_y),
        (true_value, effect_y),
        arrowstyle="<->",
        mutation_scale=10,
        linewidth=1.35,
        color=EFFECT_COLOR,
    )
    ax.add_patch(effect_arrow)
    ax.text(
        (zero_value + true_value) / 2,
        effect_y + 0.13,
        f"paired effect  +{float(new17['delta']):.5f}  ({float(new17['delta']) / zero_value:.1%})",
        ha="center",
        va="bottom",
        color=EFFECT_COLOR,
        fontsize=9,
        fontweight="bold",
    )

    ax.axvline(values[2], color=to_rgba(REFERENCE_COLOR, 0.75), linewidth=1.15, linestyle=(0, (4, 3)))

    ax.set_xlim(x_min, 0.355)
    ax.set_ylim(-0.55, 2.95)
    ax.set_yticks(y_positions, labels)
    ax.set_xticks((0.15, 0.20, 0.25, 0.30, 0.35))
    ax.set_xlabel("New 17 COCO box AP", labelpad=9)
    ax.set_title("New-class reference context", loc="left", fontsize=14.5, pad=38, fontweight="bold")
    ax.text(
        0.0,
        1.075,
        "P1-2 paired conditions shown against the released adapter-off reference",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=MUTED_COLOR,
    )
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    fig.text(
        0.5,
        0.035,
        "Fixed 65-class vocabulary · COCO val2017",
        ha="center",
        fontsize=8.6,
        color=MUTED_COLOR,
    )
    outputs = save_figure(fig, output_dir / "b1_p1_2_text_causal_new17_context")
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    rows = read_results(args.results_csv)
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    outputs.extend(plot_quality(rows, args.output_dir))
    outputs.extend(plot_new17_context(rows, args.output_dir))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
