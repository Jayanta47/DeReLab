"""
utils/visualization/model_accuracy_plots.py

Model-level accuracy comparison for the reseeding experiment — no
chain-length decomposition.  Two output files:

  plot_model_accuracy_default(df, output_path)
      One horizontal box plot: accuracy distribution across 20 seeds,
      one box per model, for default_reasoning_easy.

  plot_model_accuracy_inheritance(df, output_path)
      Same layout for linear_inheritance_easy.

Aesthetics mirror plot_depth_accuracy_boxplot in reseeding_plots.py:
same rc_context overrides, flier style, grid, and color palette.

`df` is the DataFrame returned by evaluate.analyze_reseeding.load_reseeding_results().
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

_MODEL_SHORT: dict[str, str] = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct":    "Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct":   "Llama-3.1-70B",
    "google/gemma-4-E4B":                        "Gemma-4-E4B",
    "google/gemma-4-12B-it":                     "Gemma-4-12B",
    "google/gemma-4-31B-it":                     "Gemma-4-31B",
    "google/gemma-4-26B-A4B-it":                 "Gemma-4-MoE",
    "gpt-5.1":                                   "GPT-5.1",
    "gpt-5-mini":                                "GPT-5-mini*",
    "Qwen/Qwen3-32B":                            "Qwen3-32B",
    "Qwen/Qwen3-32B__qwen3":                     "Qwen3-32B",
    "Qwen/Qwen3-32B__qwen3_thinking":            "Qwen3-32B*",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B":    "DeepSeek-R1-8B",
}

_MODEL_ORDER: list[str] = [
    "Llama-3.1-8B", "Llama-3.1-70B", "Gemma-4-E4B", "Gemma-4-12B", 
    "Gemma-4-31B", "Gemma-4-MoE",
    "GPT-5-mini*", "GPT-5.1",
    "Qwen3-32B", "Qwen3-32B*",
]

_PALETTE = [
    "#2196F3",  # blue
    "#FF5722",  # deep orange
    "#4CAF50",  # green
    "#9C27B0",  # purple
    "#FF9800",  # amber
    "#00BCD4",  # cyan
    "#F44336",  # red
    "#795548",  # brown
    "#607D8B",  # blue-grey
    "#FF7043",  # deep-orange
    "#26A69A",  # teal
]


def _short(model_id: str) -> str:
    return _MODEL_SHORT.get(model_id, model_id.split("/")[-1])


def _ordered_models(names: list[str]) -> list[str]:
    order = {m: i for i, m in enumerate(_MODEL_ORDER)}
    return sorted(names, key=lambda m: order.get(m, len(_MODEL_ORDER)))


def _make_palette(models: list[str]) -> dict[str, str]:
    ordered = _ordered_models(models)
    return {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(ordered)}


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {path}")


def _break_marks(ax_l: plt.Axes, ax_r: plt.Axes,
                 d: float = 0.030, color: str = "#555555") -> None:
    """Diagonal // marks at the break boundary; hides touching spines."""
    for ax, x in [(ax_l, 1), (ax_r, 0)]:
        kw = dict(transform=ax.transAxes, color=color,
                  clip_on=False, lw=1.6, zorder=10)
        ax.plot([x - d, x + d], [0 - d, 0 + d], **kw)
        ax.plot([x - d, x + d], [1 - d, 1 + d], **kw)
    ax_l.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_r.tick_params(which="both", left=False, labelleft=False)


def _draw_model_boxplot(
    df: pd.DataFrame,
    output_path: str,
    topology: str,
    difficulty: str,
    title: str,
    acc_col: str,
    font_size: int,
    font_color: str,
    font_family: str,
    figsize: tuple,
    x_break: float = 0.40,   # where the broken axis starts on the right panel
) -> None:
    sub = df[(df["topology"] == topology) & (df["difficulty"] == difficulty)].copy()
    if sub.empty:
        print(f"  [Skip] model_accuracy_boxplot — no data for {topology}/{difficulty}")
        return

    sub["model_name"] = sub["model_id"].apply(_short)

    # Sort models by _MODEL_ORDER; reverse so highest-ranked appears at top
    models   = list(reversed(_ordered_models(list(sub["model_name"].unique()))))
    n_models = len(models)
    palette  = _make_palette(list(sub["model_name"].unique()))

    # Resolve font family
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    if font_family not in available_fonts:
        print(f"  [Warn] Font '{font_family}' not found; falling back to 'Times'.")
        font_family = "Times"

    rc_overrides = {
        "font.family":     "serif",
        "font.serif":      [font_family, "Times New Roman", "Times", "DejaVu Serif"],
        "text.color":      font_color,
        "axes.labelcolor": font_color,
        "xtick.color":     font_color,
        "ytick.color":     font_color,
        "axes.edgecolor":  font_color,
    }
    flier_kw = dict(
        marker=".", markersize=4, alpha=0.8,
        markerfacecolor="#888888", markeredgewidth=0,
    )

    with matplotlib.rc_context(rc_overrides):
        auto_h = max(figsize[1], 0.55 * n_models + 1.5)

        # Two panels: narrow 0% stub (left) + main 40–100% range (right)
        fig, (ax_l, ax_r) = plt.subplots(
            1, 2, sharey=True, figsize=(figsize[0], auto_h),
            gridspec_kw={"width_ratios": [1, 11], "wspace": 0.04},
            layout="constrained",
        )

        bp_kw = dict(
            data=sub,
            x=acc_col,
            y="model_name",
            hue="model_name",
            order=models,
            hue_order=models,
            palette=palette,
            width=0.6,
            linewidth=1.5,
            legend=False,
            flierprops=flier_kw,
        )

        # Draw the boxplot on the right (main) panel
        sns.boxplot(**bp_kw, ax=ax_r)

        # Left stub: shows just the 0 % anchor
        ax_l.set_xlim(-0.02, 0.06)
        ax_l.set_xticks([0.0])
        ax_l.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax_l.spines["top"].set_visible(False)
        ax_l.tick_params(axis="x", labelsize=font_size, colors=font_color)
        ax_l.tick_params(axis="y", labelsize=font_size, colors=font_color)

        # Right panel: main data range
        ax_r.set_xlim(x_break - 0.02, 1.02)
        ax_r.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["right"].set_visible(False)
        ax_r.tick_params(axis="x", labelsize=font_size, colors=font_color)
        ax_r.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45, color="#aaaaaa")
        ax_r.set_axisbelow(True)

        # Shared y-label on the left panel
        ax_l.set_ylabel("Model", fontsize=font_size + 2, color=font_color)
        ax_r.set_ylabel("")

        # x-label on the main panel (conventional for broken-axis plots)
        ax_r.set_xlabel("Accuracy", fontsize=font_size + 2, color=font_color)

        _break_marks(ax_l, ax_r, color=font_color)

        fig.suptitle(title, fontsize=font_size + 4, fontweight="bold",
                     color=font_color)

    _save(fig, output_path)


def plot_model_accuracy_default(
    df: pd.DataFrame,
    output_path: str,
    acc_col: str = "overall_accuracy",
    font_size: int = 14,
    font_color: str = "#1a1a2e",
    font_family: str = "Times New Roman",
    figsize: tuple = (8, 5),
) -> None:
    """
    Horizontal box plot: model accuracy distribution across seeds,
    for default_reasoning (easy difficulty).
    """
    _draw_model_boxplot(
        df, output_path,
        topology="default_reasoning", difficulty="easy",
        title="Model Accuracy Variance — Default Reasoning",
        acc_col=acc_col,
        font_size=font_size, font_color=font_color,
        font_family=font_family, figsize=figsize,
    )


def plot_model_accuracy_inheritance(
    df: pd.DataFrame,
    output_path: str,
    acc_col: str = "overall_accuracy",
    font_size: int = 14,
    font_color: str = "#1a1a2e",
    font_family: str = "Times New Roman",
    figsize: tuple = (8, 5),
) -> None:
    """
    Horizontal box plot: model accuracy distribution across seeds,
    for linear_inheritance (easy difficulty).
    """
    _draw_model_boxplot(
        df, output_path,
        topology="linear_inheritance", difficulty="easy",
        title="Model Accuracy Variance — Linear Inheritance",
        acc_col=acc_col,
        font_size=font_size, font_color=font_color,
        font_family=font_family, figsize=figsize,
    )
