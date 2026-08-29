#!/usr/bin/env python3
"""Generate B1 YOLOE-26n P0 figures from run receipts and training logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#2F6B9A", "#E07A5F", "#59A14F")
SUBSETS = (
    ("overall65", "Overall 65"),
    ("base48", "Base 48"),
    ("new17_zero_positive_label_frozen65", "New 17"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-receipt", type=Path, required=True)
    parser.add_argument("--verify-receipt", type=Path, required=True)
    parser.add_argument("--eval-receipt", type=Path, required=True)
    parser.add_argument("--training-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "PASS":
        raise ValueError(f"Expected PASS receipt: {path}")
    return payload


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#59636E",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D8DEE5",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "svg.hashsalt": "b1-yoloe26-p0",
            "xtick.color": "#38424C",
            "ytick.color": "#38424C",
            "text.color": "#25313C",
        }
    )


def save_figure(fig: plt.Figure, png_output: Path, svg_output: Path) -> None:
    fig.savefig(png_output, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_output, bbox_inches="tight", facecolor="white", metadata={"Date": None})
    svg_text = svg_output.read_text(encoding="utf-8")
    normalized_svg = "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
    svg_output.write_text(normalized_svg, encoding="utf-8")


def label_vertical_bars(ax: plt.Axes, bars: Any, precision: int = 3) -> None:
    for bar in bars:
        value = float(bar.get_height())
        y = value + 0.009 if value > 0 else 0.009
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.{precision}f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if value > 0 else "normal",
        )


def plot_quality(eval_receipt: dict[str, Any], output_dir: Path) -> list[Path]:
    metrics = eval_receipt["metrics"]
    panels = (
        ("COCO box metrics", ("AP", "AP50", "AP75")),
        ("AP by object size", ("APs", "APm", "APl")),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.25))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.16, wspace=0.12)
    x = np.arange(len(panels[0][1]))
    width = 0.24

    for ax, (title, metric_names) in zip(axes, panels, strict=True):
        for index, ((subset_key, subset_label), color) in enumerate(zip(SUBSETS, COLORS, strict=True)):
            values = [float(metrics[subset_key][name]) for name in metric_names]
            bars = ax.bar(
                x + (index - 1) * width,
                values,
                width,
                label=subset_label,
                color=color,
                edgecolor="white",
                linewidth=0.7,
            )
            label_vertical_bars(ax, bars)

        ax.set_title(title, loc="left")
        ax.set_xticks(x, metric_names)
        ax.set_ylabel("Average precision")
        ax.set_ylim(0, 0.56)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.86), ncols=3)
    fig.suptitle("B1 YOLOE-26n P0 · COCO 48/17 Evaluation", fontsize=16, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.035,
        "Source: INFERENCE_EVAL_RECEIPT.json · 5,000 COCO val2017 images · bbox COCOeval",
        ha="center",
        fontsize=9,
        color="#59636E",
    )

    png_output = output_dir / "b1_yoloe26_p0_quality.png"
    svg_output = output_dir / "b1_yoloe26_p0_quality.svg"
    save_figure(fig, png_output, svg_output)
    plt.close(fig)
    return [png_output, svg_output]


def human_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    return f"{seconds:.2f} s"


def plot_execution(
    main_receipt: dict[str, Any],
    verify_receipt: dict[str, Any],
    eval_receipt: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    stage_names = ("Training\n80 epochs", "Checkpoint\nverification", "Independent\nCOCO evaluation")
    stage_seconds = np.array(
        [
            float(main_receipt["wall_time_seconds"]),
            float(verify_receipt["wall_time_seconds"]),
            float(eval_receipt["wall_time_seconds"]),
        ]
    )

    runtime = eval_receipt["runtime"]
    latency = runtime["per_result_speed_latency_ms"]
    quantile_names = ("P50", "P95", "P99")
    quantile_values = np.array([float(latency[name.lower()]) for name in quantile_names])
    throughput = float(runtime["throughput_images_per_second"])
    peak_gib = float(runtime["peak_gpu_memory_allocated_bytes"]) / (1024**3)
    image_count = int(eval_receipt["counts"]["overall65"]["image_count"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    ax = axes[0]
    y = np.arange(len(stage_names))
    bars = ax.barh(y, stage_seconds, color=COLORS, height=0.55)
    ax.set_xscale("log")
    ax.set_xlim(1, 200000)
    ax.set_yticks(y, stage_names)
    ax.invert_yaxis()
    ax.set_xlabel("Execution time (seconds, log scale)")
    ax.set_title("Completed stages", loc="left")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, seconds in zip(bars, stage_seconds, strict=True):
        ax.text(
            seconds * 1.12,
            bar.get_y() + bar.get_height() / 2,
            human_duration(float(seconds)),
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    ax = axes[1]
    latency_bars = ax.bar(quantile_names, quantile_values, color=COLORS, width=0.58)
    ax.set_ylim(0, max(quantile_values) * 1.28)
    ax.set_ylabel("Latency per image (ms)")
    ax.set_title("Independent inference profile", loc="left")
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    label_vertical_bars(ax, latency_bars, precision=2)
    ax.text(
        0.5,
        0.94,
        f"{image_count:,} images  ·  {throughput:.2f} images/s  ·  {peak_gib:.2f} GiB peak allocated",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#59636E",
    )

    fig.suptitle("B1 YOLOE-26n P0 · Execution Profile", fontsize=16, fontweight="bold", y=1.04)
    fig.text(
        0.5,
        -0.035,
        "Sources: MAIN_RECEIPT.json · VERIFY_RECEIPT.json · INFERENCE_EVAL_RECEIPT.json",
        ha="center",
        fontsize=9,
        color="#59636E",
    )

    png_output = output_dir / "b1_yoloe26_p0_execution.png"
    svg_output = output_dir / "b1_yoloe26_p0_execution.svg"
    save_figure(fig, png_output, svg_output)
    plt.close(fig)
    return [png_output, svg_output]


def read_training_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw_row in reader:
            row = {key.strip(): float(value) for key, value in raw_row.items() if key and value}
            rows.append(row)
    if not rows:
        raise ValueError(f"Training CSV is empty: {path}")
    return rows


def plot_training_curves(path: Path, output_dir: Path) -> list[Path]:
    rows = read_training_csv(path)
    required = (
        "epoch",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    )
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise ValueError(f"Training CSV is missing columns: {', '.join(missing)}")

    epochs = np.array([row["epoch"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    loss_names = (("train/box_loss", "Box loss"), ("train/cls_loss", "Class loss"), ("train/dfl_loss", "DFL loss"))
    for (key, label), color in zip(loss_names, COLORS, strict=True):
        axes[0].plot(epochs, [row[key] for row in rows], label=label, color=color, linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_title("Training losses (log scale)", loc="left")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_xlim(1, 80)
    axes[0].set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80])
    axes[0].legend()

    metric_names = (
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50–95"),
    )
    metric_colors = COLORS + ("#B07AA1",)
    for (key, label), color in zip(metric_names, metric_colors, strict=True):
        values = [row[key] for row in rows]
        axes[1].plot(epochs, values, color=color, linewidth=2)
        axes[1].scatter(epochs[-1], values[-1], color=color, s=22, zorder=3)
        axes[1].text(
            81.5,
            values[-1],
            f"{label}  {values[-1]:.3f}",
            color=color,
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[1].set_title("Validation metrics", loc="left")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_xlim(1, 103)
    axes[1].set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80])

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True)

    fig.suptitle("B1 YOLOE-26n P0 · Training Curves", fontsize=16, fontweight="bold", y=1.04)
    fig.text(
        0.5,
        -0.035,
        f"Source: {path.name} · {len(rows)} completed epochs",
        ha="center",
        fontsize=9,
        color="#59636E",
    )
    png_output = output_dir / "b1_yoloe26_p0_training.png"
    svg_output = output_dir / "b1_yoloe26_p0_training.svg"
    save_figure(fig, png_output, svg_output)
    plt.close(fig)
    return [png_output, svg_output]


def main() -> None:
    args = parse_args()
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main_receipt = load_json(args.main_receipt)
    verify_receipt = load_json(args.verify_receipt)
    eval_receipt = load_json(args.eval_receipt)

    outputs = []
    outputs.extend(plot_quality(eval_receipt, args.output_dir))
    outputs.extend(plot_execution(main_receipt, verify_receipt, eval_receipt, args.output_dir))
    if args.training_csv:
        outputs.extend(plot_training_curves(args.training_csv, args.output_dir))

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
