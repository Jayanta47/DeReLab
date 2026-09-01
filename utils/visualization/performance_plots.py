"""
utils/visualization/performance_plots.py

Three publication-ready figures for the main performance analysis.

plot_heatmap(summary_df, output_path)
    Model × (Topology × Difficulty) heatmap.  Only models with all 6 result
    files are included.  No colour gradient — cells are white, values are
    coloured by rank within each column.  Best per column is highlighted.

plot_depth_lines(depth_df, output_path)
    Accuracy vs. reasoning-complexity proxy, one panel per topology.
      default_reasoning   → chain length
      linear_inheritance  → hierarchy depth, with a broken x-axis that
                            elides the gap between easy (4–6) and hard (20–30)
      tree_inheritance    → graph size (# class nodes), binned into 6 groups
                            per difficulty
    Only models with all 6 result files are plotted.

render_detail_table / export_latex_table  — appendix table (unchanged API).
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

matplotlib.rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":           16,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "figure.dpi":          150,
    "axes.labelsize":      16,
    "axes.titlesize":      16,
    "xtick.labelsize":     16,
    "ytick.labelsize":     18,
    "legend.fontsize":     16,
})

# ── shared constants ──────────────────────────────────────────────────────────

_PALETTE = [
    "#2196F3",  # blue
    "#E53935",  # red
    "#43A047",  # green
    "#8E24AA",  # purple
    "#FB8C00",  # orange
    "#00ACC1",  # cyan
    "#F06292",  # pink
    "#6D4C41",  # brown
]

_MODEL_ORDER = [
    "Llama-3.1-8B", "Gemma-4-E4B", "Gemma-4-12B",
    "Llama-3.1-70B", "Gemma-4-31B", "Gemma-4-MoE",
    "GPT-5-mini*", "GPT-5.1",
    "Qwen3-32B", "Qwen3-32B*", "DeepSeek-R1-8B",
]

_CONFIG_ORDER = [
    "default_reasoning_easy_entailment", "default_reasoning_hard_entailment",
    "default_reasoning_easy_effect",     "default_reasoning_hard_effect",
    "linear_inheritance_easy",           "linear_inheritance_hard",
    "tree_inheritance_easy",             "tree_inheritance_hard",
]

_TOPO_LABEL = {
    "default_reasoning":  "Default Reasoning",
    "linear_inheritance": "Linear Inheritance",
    "tree_inheritance":   "Tree Inheritance",
}

_DEPTH_XLABEL = {
    "default_reasoning":  "Chain Length",
    "linear_inheritance": "Hierarchy Depth",
    "tree_inheritance":   "Graph Size (# Class Nodes)",
}

# Topology group boundaries (inclusive column indices in _CONFIG_ORDER)
_TOPO_SPANS = [
    (0, 3, "Default Reasoning"),
    (4, 5, "Linear Inheritance"),
    (6, 7, "Tree Inheritance"),
]


def _complete_models(summary_df: pd.DataFrame) -> set[str]:
    counts = summary_df.groupby("model_name").size()
    return set(counts[counts >= 1].index)


def _model_palette(models: list[str]) -> dict[str, str]:
    ordered = [m for m in _MODEL_ORDER if m in models] + \
              [m for m in models if m not in _MODEL_ORDER]
    return {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(ordered)}


def _sort_models(models: list[str]) -> list[str]:
    order = {m: i for i, m in enumerate(_MODEL_ORDER)}
    return sorted(models, key=lambda m: order.get(m, len(_MODEL_ORDER)))


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Heatmap  (viridis gradient, wide & compressed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(
    summary_df: pd.DataFrame,
    output_path: str,
    figsize: Optional[tuple] = None,
    cmap: str = "Blues",
    sort_by_accuracy: bool = True,
) -> None:
    """
    Wide, vertically-compressed heatmap using a viridis colour gradient.

    • Only models with all 6 result files included, sorted best-to-worst.
    • Two-line column labels combine topology and difficulty so nothing is
      obscured.
    • Thick white dividers separate the three topology groups.
    • Annotation text colour adapts to cell luminance (white on dark cells,
      black on light cells) for readability across the full viridis range.
    """
    complete = _complete_models(summary_df)
    df = summary_df[summary_df["model_name"].isin(complete)].copy()

    # ── expand default_reasoning into entailment + effect columns ─────────────
    # All other topologies stay as a single overall_accuracy column.
    expanded_rows = []
    for _, r in df.iterrows():
        topo, diff = r["topology"], r["difficulty"]
        if topo == "default_reasoning":
            expanded_rows.append({
                "model_name": r["model_name"],
                "config": f"default_reasoning_{diff}_entailment",
                "accuracy": r["acc_entailment"],
            })
            expanded_rows.append({
                "model_name": r["model_name"],
                "config": f"default_reasoning_{diff}_effect",
                "accuracy": r["acc_effect_belief"],
            })
        else:
            expanded_rows.append({
                "model_name": r["model_name"],
                "config": r["config"],
                "accuracy": r["overall_accuracy"],
            })
    expanded_df = pd.DataFrame(expanded_rows)

    # ── pivot ─────────────────────────────────────────────────────────────────
    pivot = (
        expanded_df.groupby(["model_name", "config"])["accuracy"]
        .mean().reset_index()
        .pivot(index="model_name", columns="config", values="accuracy")
    )
    for cfg in _CONFIG_ORDER:
        if cfg not in pivot.columns:
            pivot[cfg] = np.nan
    pivot = pivot[_CONFIG_ORDER]
    if sort_by_accuracy:
        row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    else:
        row_order = [m for m in _MODEL_ORDER if m in pivot.index] + \
                    [m for m in pivot.index if m not in _MODEL_ORDER]
    pivot = pivot.reindex(row_order)

    # ── average column ────────────────────────────────────────────────────────
    pivot["avg"] = pivot.mean(axis=1)

    n_rows, n_cols = pivot.shape   # now 9 columns
    vmin, vmax = 0.1, 1.0

    # ── annotation strings ────────────────────────────────────────────────────
    annot = np.array([
        [f"{v * 100:.1f}" if not np.isnan(v) else "—" for v in row]
        for row in pivot.values
    ])

    # ── figure ────────────────────────────────────────────────────────────────
    fig_w = figsize[0] if figsize else 16
    fig_h = figsize[1] if figsize else max(2.6, n_rows * 0.42 + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── draw heatmap ──────────────────────────────────────────────────────────
    cm = plt.get_cmap(cmap)
    cm_obj = matplotlib.colors.Colormap(cmap) if hasattr(matplotlib.colors, 'Colormap') else cm
    norm   = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        linewidths=2.0,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.75, "pad": 0.02, "aspect": 25},
        annot_kws={"size": 13, "weight": "bold"},
    )

    # ── adapt annotation text colour to cell luminance ────────────────────────
    # seaborn draws Text objects for each annotation — recolour them.
    for text_obj in ax.texts:
        try:
            txt = text_obj.get_text()
            if txt == "—":
                text_obj.set_color("#888888")
                continue
            val = float(txt) / 100.0
            rgba  = cm(norm(val))
            lum   = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_obj.set_color("white" if lum < 0.45 else "#111111")
        except (ValueError, TypeError):
            pass

    # ── column labels ─────────────────────────────────────────────────────────
    col_labels = [
        "Entail\nEasy",  "Entail\nHard",
        "Effect\nEasy",  "Effect\nHard",
        "Easy",          "Hard",
        "Easy",          "Hard",
        "Avg",
    ]
    ax.set_xticklabels(col_labels, rotation=0, ha="center",
                       fontsize=12, va="top")
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="x", which="both", bottom=False, pad=4)

    # ── topology group separators ─────────────────────────────────────────────
    for x_sep in [4.0, 6.0, 8.0]:      # 8.0 separates avg from the rest
        ax.axvline(x=x_sep, color="white", linewidth=5, zorder=4)
    ax.axvline(x=2.0, color="white", linewidth=2.5, zorder=4)

    # ── row labels ────────────────────────────────────────────────────────────
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="y", which="both", left=False, pad=5)

    # ── topology group bracket labels above the heatmap (single row) ──────────
    # 9 columns total: fractions based on 9
    for x0f, x1f, name in [
        (0,    4/9, "Default Reasoning"),
        (4/9,  6/9, "Linear Inheritance"),
        (6/9,  8/9, "Tree Inheritance"),
    ]:
        xc = (x0f + x1f) / 2
        ax.text(xc, 1.06, name, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                color="#1A237E")
        pad = 0.008
        ax.annotate("", xy=(x1f - pad, 1.02), xytext=(x0f + pad, 1.02),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", color="#1A237E",
                                   lw=1.2, connectionstyle="arc3,rad=0"))

    # ── colour-bar label ──────────────────────────────────────────────────────
    cbar = ax.collections[0].colorbar
    cbar.set_label("Overall Accuracy", fontsize=12)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    cbar.ax.tick_params(labelsize=11)

    ax.set_title(
        "Model Performance — Overall Accuracy by Topology and Difficulty",
        fontsize=20, fontweight="bold", pad=38, color="#000000",
    )

    fig.tight_layout(pad=1.2)
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Accuracy vs. Reasoning Depth  (broken x-axis for linear inheritance)
# ─────────────────────────────────────────────────────────────────────────────

_N_TREE_BINS = 6   # quantile bins for tree graph-size axis


def _bin_depth(series: pd.Series, n_bins: int = _N_TREE_BINS) -> pd.Series:
    """Return bin-midpoint values for quantile binning."""
    try:
        cut = pd.qcut(series, q=n_bins, duplicates="drop")
        return cut.apply(lambda b: round((b.left + b.right) / 2, 1))
    except Exception:
        return series


def _broken_x_marks(ax_l: plt.Axes, ax_r: plt.Axes, d: float = 0.025) -> None:
    """
    Draw diagonal break marks ( // ) at the boundary between two adjacent axes
    and hide the touching spines so the break looks seamless.
    """
    for ax, x in [(ax_l, 1), (ax_r, 0)]:
        kw = dict(transform=ax.transAxes, color="#555555",
                  clip_on=False, lw=1.6, zorder=10)
        ax.plot([x - d, x + d], [0 - d, 0 + d], **kw)
        ax.plot([x - d, x + d], [1 - d, 1 + d], **kw)

    ax_l.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_r.tick_params(which="both", left=False, labelleft=False)


def plot_depth_lines(
    depth_df: pd.DataFrame,
    output_path: str,
    figsize: Optional[tuple] = None,
) -> None:
    """
    Three-panel line plot: Default | Linear (broken-axis) | Tree.

    • Only models with ≥ 6 result files are plotted.
    • Linear inheritance: broken x-axis that elides the gap between easy
      chain lengths (4–6) and hard chain lengths (20–30).
    • Tree inheritance: x-axis is graph size (# class nodes), binned into
      up to 6 quantile groups per difficulty so the line is smooth.
    • Solid lines = easy, dashed lines = hard.
    • Shaded bands = ±1 SEM.
    """
    if depth_df.empty:
        print("[Plot] depth_df is empty — skipping depth line plot.")
        return

    # ── filter to complete models only ───────────────────────────────────────
    # Build a temporary summary to find complete models
    complete = set(
        depth_df.groupby("model_name").apply(
            lambda g: len(set(zip(g["topology"], g["difficulty"])))
        ).pipe(lambda s: s[s >= 1].index)
    )
    df = depth_df[depth_df["model_name"].isin(complete)].copy()
    if df.empty:
        print("[Plot] No complete models found — plotting all available.")
        df = depth_df.copy()

    all_models = _sort_models(df["model_name"].unique().tolist())
    palette    = _model_palette(all_models)
    ls_map     = {"easy": "-", "hard": "--"}
    diff_label = {"easy": "Easy", "hard": "Hard"}

    # ── bin tree x-axis ───────────────────────────────────────────────────────
    df["depth_val"] = df["depth_val"].astype(float)
    tree_mask = df["topology"] == "tree_inheritance"
    if tree_mask.any():
        binned = (
            df.loc[tree_mask]
            .groupby("difficulty", group_keys=False)["depth_val"]
            .transform(_bin_depth)
        )
        df.loc[tree_mask, "depth_val"] = binned.values
        # Re-aggregate so each (model, topo, diff, bin) is one point
        df = (
            df.groupby(["model_id", "model_name", "topology", "difficulty",
                        "depth_val", "depth_label"])
            .agg(
                mean_accuracy=("mean_accuracy", "mean"),
                sem_accuracy=("sem_accuracy", "mean"),
                n_conversations=("n_conversations", "sum"),
            )
            .reset_index()
        )

    # ── figure layout ─────────────────────────────────────────────────────────
    fig_w = figsize[0] if figsize else 26.0
    fig_h = figsize[1] if figsize else 4
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Three top-level panels: Default | Linear | Tree
    outer = GridSpec(1, 3, figure=fig, wspace=0.36, width_ratios=[1, 1, 1])

    ax_default = fig.add_subplot(outer[0])

    # Linear splits into easy sub-panel (narrow) + hard sub-panel (wider)
    inner_lin = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        wspace=0.06, width_ratios=[1, 2],
    )
    ax_lin_l = fig.add_subplot(inner_lin[0])
    ax_lin_r = fig.add_subplot(inner_lin[1], sharey=ax_lin_l)

    ax_tree = fig.add_subplot(outer[2])

    # ── helper: plot one topology panel ──────────────────────────────────────
    def _plot_panel(ax, topo, diff_filter=None):
        sub = df[df["topology"] == topo]
        if diff_filter:
            sub = sub[sub["difficulty"] == diff_filter]
        if sub.empty:
            return
        for model in all_models:
            color = palette.get(model, "#888888")
            for diff, ls in ls_map.items():
                if diff_filter and diff != diff_filter:
                    continue
                ms = sub[(sub["model_name"] == model) & (sub["difficulty"] == diff)]
                if ms.empty:
                    continue
                ms = ms.sort_values("depth_val")
                xs, ys, sems = ms["depth_val"].values, ms["mean_accuracy"].values, ms["sem_accuracy"].values
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2.0,
                        marker="o", markersize=5, markeredgewidth=0.5,
                        markeredgecolor="white")
                ax.fill_between(xs, ys - sems, ys + sems, color=color, alpha=0.12)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
        ax.grid(axis="both", linestyle="--", alpha=0.30, color="#B0BEC5")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # Default panel
    _plot_panel(ax_default, "default_reasoning")
    ax_default.set_title("Default Reasoning", fontsize=13, fontweight="bold", pad=6)
    ax_default.set_xlabel(_DEPTH_XLABEL["default_reasoning"], fontsize=12)
    ax_default.set_ylabel("Accuracy", fontsize=12)

    # Linear panels (broken axis)
    _plot_panel(ax_lin_l, "linear_inheritance", "easy")
    _plot_panel(ax_lin_r, "linear_inheritance", "hard")
    # Shared y-axis: only left panel shows y-label/ticks
    ax_lin_l.set_ylabel("Accuracy", fontsize=12)
    # x-axis limits snug around the data
    lin_easy_vals = df.loc[(df.topology == "linear_inheritance") & (df.difficulty == "easy"), "depth_val"]
    lin_hard_vals = df.loc[(df.topology == "linear_inheritance") & (df.difficulty == "hard"), "depth_val"]
    if not lin_easy_vals.empty:
        margin = 0.5
        ax_lin_l.set_xlim(lin_easy_vals.min() - margin, lin_easy_vals.max() + margin)
    if not lin_hard_vals.empty:
        margin = 1.0
        ax_lin_r.set_xlim(lin_hard_vals.min() - margin, lin_hard_vals.max() + margin)
    _broken_x_marks(ax_lin_l, ax_lin_r)

    # Shared x-label centred across both linear sub-panels
    fig.text(
        (ax_lin_l.get_position().x0 + ax_lin_r.get_position().x1) / 2,
        ax_lin_l.get_position().y0 - 0.065,
        _DEPTH_XLABEL["linear_inheritance"],
        ha="center", va="top", fontsize=12,
    )
    # "Linear Inheritance" title centred across both sub-panels
    mid_x = (ax_lin_l.get_position().x0 + ax_lin_r.get_position().x1) / 2
    top_y = max(ax_lin_l.get_position().y1, ax_lin_r.get_position().y1)
    fig.text(mid_x, top_y + 0.04, "Linear Inheritance",
             ha="center", va="bottom", fontsize=13, fontweight="bold")

    # Tree panel
    _plot_panel(ax_tree, "tree_inheritance")
    ax_tree.set_title("Tree Inheritance", fontsize=13, fontweight="bold", pad=6)
    ax_tree.set_xlabel(_DEPTH_XLABEL["tree_inheritance"], fontsize=12)
    ax_tree.set_ylabel("Accuracy", fontsize=12)
    ax_tree.xaxis.set_major_locator(mticker.MaxNLocator(integer=False, nbins=5))

    # ── legend ────────────────────────────────────────────────────────────────
    model_handles = [
        mlines.Line2D([], [], color=palette[m], linewidth=2.2,
                      marker="o", markersize=5, label=m)
        for m in all_models if m in palette
    ]
    diff_handles = [
        mlines.Line2D([], [], color="gray", linestyle="-",  linewidth=2, label="Easy"),
        mlines.Line2D([], [], color="gray", linestyle="--", linewidth=2, label="Hard"),
    ]
    sep = mlines.Line2D([], [], color="none", label="")

    fig.legend(
        handles=model_handles + [sep] + diff_handles,
        loc="lower center", ncol=len(model_handles) + 3,
        bbox_to_anchor=(0.5, -0.13),
        fontsize=11, framealpha=0.95, edgecolor="#CFD8DC",
        handlelength=2.0,
    )

    fig.suptitle("Accuracy vs. Reasoning Complexity",
                 fontsize=15, fontweight="bold", y=1.03)
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Detailed appendix table (matplotlib figure)
# ─────────────────────────────────────────────────────────────────────────────

def render_detail_table(
    detail_df: pd.DataFrame,
    output_path: str,
    figsize: Optional[tuple] = None,
) -> None:
    if detail_df.empty:
        print("[Plot] detail_df is empty — skipping appendix table.")
        return

    topo_nice = {
        "default_reasoning":  "Default",
        "linear_inheritance": "Linear",
        "tree_inheritance":   "Tree",
    }

    def _pct(val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        return f"{val * 100:.1f}%"

    def _cnt(val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        return str(int(val))

    col_headers = [
        "Model", "Topology", "Diff.", "N",
        "Overall", "Init. Acc", "Init. N",
        "Update Acc", "Update N",
        "Effect Acc", "Effect N",
        "Hyp. Chg. Acc", "Hyp. Chg. N",
    ]

    table_data = []
    for _, row in detail_df.iterrows():
        table_data.append([
            row["model_name"],
            topo_nice.get(row["topology"], row["topology"]),
            row["difficulty"].capitalize(),
            _cnt(row.get("n_samples")),
            _pct(row.get("overall_accuracy")),
            _pct(row.get("acc_initial_answer")),
            _cnt(row.get("n_initial_answer")),
            _pct(row.get("acc_update_answer")),
            _cnt(row.get("n_update_answer")),
            _pct(row.get("acc_effect_of_update")),
            _cnt(row.get("n_effect_of_update")),
            _pct(row.get("acc_hypothesis_change")),
            _cnt(row.get("n_hypothesis_change")),
        ])

    nrows = len(table_data)
    ncols = len(col_headers)
    col_widths = [1.4, 0.85, 0.6, 0.5,
                  0.75, 0.75, 0.6, 0.85, 0.75,
                  0.75, 0.75, 1.1, 0.95]
    total_w = sum(col_widths)

    fig_w = figsize[0] if figsize else min(max(total_w + 0.5, 14), 22)
    row_h = 0.38
    fig_h = figsize[1] if figsize else max(4.0, (nrows + 2) * row_h + 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="upper center",
        colWidths=[w / total_w for w in col_widths],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    for j in range(ncols):
        cell = tbl[0, j]
        cell.set_facecolor("#1A237E")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)

    topo_bg = {"Default": "#EAF4FB", "Linear": "#FDFEFE", "Tree": "#F0FFF0"}
    for i, row_data in enumerate(table_data):
        topo_key = row_data[1]
        base_bg = topo_bg.get(topo_key, "white")
        shade = "#D6EAF8" if (i % 2 == 0 and topo_key == "Default") else \
                "#F2F3F4" if (i % 2 == 0 and topo_key == "Linear")  else \
                "#D5F5E3" if (i % 2 == 0 and topo_key == "Tree")    else base_bg
        for j in range(ncols):
            tbl[i + 1, j].set_facecolor(shade)

    ax.set_title(
        "Detailed Performance Breakdown by Model, Topology, and Difficulty\n"
        "(Init. = Initial Answer; Update = Update Answer; "
        "Effect = Effect of Update; Hyp. Chg. = Hypothesis Change)",
        fontsize=12, pad=12, fontweight="bold",
    )
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. LaTeX appendix table
# ─────────────────────────────────────────────────────────────────────────────

def export_latex_table(detail_df: pd.DataFrame, output_path: str) -> None:
    if detail_df.empty:
        print("[LaTeX] detail_df is empty — skipping.")
        return

    topo_nice = {
        "default_reasoning":  "Default",
        "linear_inheritance": "Linear",
        "tree_inheritance":   "Tree",
    }

    def _pct(val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "---"
        return f"{val * 100:.1f}"

    def _cnt(val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "---"
        return str(int(val))

    lines = [
        r"\begin{longtable}{llcccccccccccc}",
        r"\caption{Detailed model performance breakdown by topology, difficulty, and task type. "
        r"Accuracies are given in \%. "
        r"``Effect'' = effect-of-update accuracy (default reasoning only); "
        r"``Hyp.\ Chg.'' = hypothesis-change accuracy (default reasoning only).}\\",
        r"\label{tab:full_results}\\",
        r"\toprule",
        r"\textbf{Model} & \textbf{Topo.} & \textbf{Diff.} & \textbf{N} "
        r"& \textbf{Overall} "
        r"& \textbf{Init.} & \textbf{Init.\ N} "
        r"& \textbf{Update} & \textbf{Upd.\ N} "
        r"& \textbf{Effect} & \textbf{Eff.\ N} "
        r"& \textbf{Hyp.\ Chg.} & \textbf{HC\ N} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Model} & \textbf{Topo.} & \textbf{Diff.} & \textbf{N} "
        r"& \textbf{Overall} "
        r"& \textbf{Init.} & \textbf{Init.\ N} "
        r"& \textbf{Update} & \textbf{Upd.\ N} "
        r"& \textbf{Effect} & \textbf{Eff.\ N} "
        r"& \textbf{Hyp.\ Chg.} & \textbf{HC\ N} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{13}{r}{\textit{continued\ldots}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for i, (_, row) in enumerate(detail_df.iterrows()):
        shade = r"\rowcolor{gray!8}" if i % 2 == 0 else ""
        model = row["model_name"].replace("-", r"\mbox{-}")
        topo  = topo_nice.get(row["topology"], row["topology"])
        diff  = row["difficulty"].capitalize()
        cells = [
            model, topo, diff, _cnt(row.get("n_samples")),
            _pct(row.get("overall_accuracy")),
            _pct(row.get("acc_initial_answer")), _cnt(row.get("n_initial_answer")),
            _pct(row.get("acc_update_answer")),  _cnt(row.get("n_update_answer")),
            _pct(row.get("acc_effect_of_update")), _cnt(row.get("n_effect_of_update")),
            _pct(row.get("acc_hypothesis_change")), _cnt(row.get("n_hypothesis_change")),
        ]
        row_str = " & ".join(cells) + r" \\"
        if shade:
            lines.append(shade)
        lines.append(row_str)

    lines.append(r"\end{longtable}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[LaTeX] Saved → {output_path}")
