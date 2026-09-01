"""
utils/visualization/reseeding_plots.py

Plotting functions for the E5 seed-stability experiment.

Four public functions
--------------------
plot_seed_strip          — beeswarm/strip plot: accuracy distribution across seeds,
                           one row per (model × config), colored by model.

plot_accuracy_vs_chain_length
                         — scatter: x = accuracy, y = mean chain length per seed,
                           colored by model, faceted by config.  Shows whether
                           seeds that produce longer chains also produce lower accuracy.

plot_accuracy_vs_model_sd
                         — scatter: x = mean accuracy, y = SD of accuracy across models,
                           one point per (seed × config).  Highlights discriminative seeds.

render_stats_table       — matplotlib figure of the ICC / KW / variance summary table
                           produced by evaluation/analyze_reseeding.py.

Usage
-----
    from utils.visualization.reseeding_plots import plot_seed_strip, render_stats_table
    plot_seed_strip(df, "outputs/figures/seed_strip.pdf")
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":      13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi":     150,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# ── palette: one color per model, consistent across all plots ──────────────
_DEFAULT_PALETTE = [
    "#2196F3",   # blue
    "#FF5722",   # deep orange
    "#4CAF50",   # green
    "#9C27B0",   # purple
    "#FF9800",   # amber
    "#00BCD4",   # cyan
    "#F44336",   # red
    "#795548",   # brown
    "#607D8B",   # blue-grey
]

_CONFIG_LABELS = {
    "default_reasoning_easy":    "Default (Easy)",
    "default_reasoning_hard":    "Default (Hard)",
    "linear_inheritance_easy":   "Linear Inh. (Easy)",
    "linear_inheritance_hard":   "Linear Inh. (Hard)",
    "tree_inheritance_easy":     "Tree Inh. (Easy)",
    "tree_inheritance_hard":     "Tree Inh. (Hard)",
}

_MODEL_SHORT = {
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

_MODEL_ORDER = [
    "Llama-3.1-8B", "Gemma-4-E4B", "Gemma-4-12B",
    "Llama-3.1-70B", "Gemma-4-31B", "Gemma-4-MoE",
    "GPT-5-mini*", "GPT-5.1",
    "Qwen3-32B", "Qwen3-32B*", "DeepSeek-R1-8B",
]


def _short(model_id: str) -> str:
    return _MODEL_SHORT.get(model_id, model_id.split("/")[-1])


def _config_label(cfg: str) -> str:
    return _CONFIG_LABELS.get(cfg, cfg)


def _model_palette(models: list[str]) -> dict:
    order = {m: i for i, m in enumerate(_MODEL_ORDER)}
    ordered = sorted(models, key=lambda m: order.get(_short(m), len(_MODEL_ORDER)))
    return {m: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)] for i, m in enumerate(ordered)}


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Seed strip plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_seed_strip(
    df: pd.DataFrame,
    output_path: str,
    task_type: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    Horizontal strip (beeswarm) plot.

    Each row on the y-axis is a (model × config) cell.
    Each dot is one seed's accuracy value.
    Overlaid box shows IQR and median.
    Models share a consistent colour across configs.

    Parameters
    ----------
    df          : DataFrame from analyze_reseeding.load_reseeding_results()
    output_path : File path for the saved figure (.pdf / .png / .svg)
    task_type   : If given, use per-task accuracy column (e.g. "update_answer").
                  If None, uses overall_accuracy.
    figsize     : (width, height) in inches.
    """
    acc_col = f"acc_{task_type}" if task_type else "overall_accuracy"
    if acc_col not in df.columns:
        raise ValueError(f"Column '{acc_col}' not found. Available: {list(df.columns)}")

    models = sorted(df["model_id"].unique())
    configs = sorted(df["config"].unique())
    palette = _model_palette(models)

    # Build row labels: "{model_short} — {config_label}"
    rows = [(m, c) for c in configs for m in models]
    row_labels = [f"{_short(m)}\n{_config_label(c)}" for m, c in rows]
    row_idx = {(m, c): i for i, (m, c) in enumerate(rows)}

    fig, ax = plt.subplots(figsize=figsize)

    for model in models:
        color = palette[model]
        for cfg in configs:
            subset = df[(df["model_id"] == model) & (df["config"] == cfg)]
            if subset.empty:
                continue
            yi = row_idx[(model, cfg)]
            accs = subset[acc_col].dropna().values

            # Box: IQR
            q25, q50, q75 = np.percentile(accs, [25, 50, 75])
            iqr = q75 - q25
            ax.barh(yi, iqr, left=q25, height=0.35, color=color, alpha=0.25, zorder=2)
            ax.vlines(q50, yi - 0.175, yi + 0.175, color=color, linewidth=2, zorder=3)

            # Jitter dots
            jitter = np.random.default_rng(42).uniform(-0.18, 0.18, size=len(accs))
            ax.scatter(accs, yi + jitter, color=color, s=28, alpha=0.7,
                       zorder=4, label=_short(model) if cfg == configs[0] else "_nolegend_")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Accuracy", fontsize=13)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(
        "Seed Stability — Accuracy Distribution Across 20 Seeds"
        + (f"\n(task: {task_type})" if task_type else ""),
        fontsize=14, pad=10,
    )
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), title="Model", fontsize=11,
              loc="lower right", framealpha=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Accuracy vs mean chain length
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_chain_length(
    df: pd.DataFrame,
    output_path: str,
    task_type: Optional[str] = None,
    figsize: tuple = (11, 5),
) -> None:
    """
    Scatter: x = accuracy, y = mean chain length for that seed's dataset.
    One panel per config, one colour per model.

    Useful for showing whether seeds that randomly generate longer chains
    also produce lower model accuracy — i.e. whether between-seed variance
    in accuracy is explained by structural difficulty.
    """
    acc_col = f"acc_{task_type}" if task_type else "overall_accuracy"
    if acc_col not in df.columns:
        raise ValueError(f"Column '{acc_col}' not found.")
    if "mean_chain_length" not in df.columns or df["mean_chain_length"].isna().all():
        raise ValueError("mean_chain_length is not available in the DataFrame.")

    models = sorted(df["model_id"].unique())
    configs = sorted(df["config"].unique())
    palette = _model_palette(models)

    ncols = len(configs)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, cfg in zip(axes, configs):
        sub = df[df["config"] == cfg].dropna(subset=[acc_col, "mean_chain_length"])
        for model in models:
            ms = sub[sub["model_id"] == model]
            if ms.empty:
                continue
            ax.scatter(
                ms[acc_col], ms["mean_chain_length"],
                color=palette[model], label=_short(model),
                s=55, alpha=0.75, edgecolors="white", linewidths=0.5,
            )
            # Trend line
            if len(ms) >= 3:
                z = np.polyfit(ms[acc_col], ms["mean_chain_length"], 1)
                xs = np.linspace(ms[acc_col].min(), ms[acc_col].max(), 50)
                ax.plot(xs, np.poly1d(z)(xs), color=palette[model],
                        linestyle="--", linewidth=1.2, alpha=0.6)

        ax.set_title(_config_label(cfg), fontsize=13)
        ax.set_xlabel("Accuracy", fontsize=12)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Mean Chain Length (per seed)", fontsize=12)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Model", fontsize=11,
               loc="lower center", ncol=len(models),
               bbox_to_anchor=(0.5, -0.06), framealpha=0.9)
    fig.suptitle(
        "Accuracy vs. Seed Chain Length"
        + (f" — task: {task_type}" if task_type else ""),
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Accuracy vs cross-model SD  (discriminability plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_model_sd(
    df: pd.DataFrame,
    output_path: str,
    task_type: Optional[str] = None,
    figsize: tuple = (10, 5),
) -> None:
    """
    Scatter: x = mean accuracy across models, y = SD of accuracy across models.
    One point per (seed × config), coloured by config.

    High SD = discriminative seed (models disagree).
    Low SD = easy or impossible seed (models agree on wrong / right).
    Useful for identifying which seeds best separate model capability.
    """
    acc_col = f"acc_{task_type}" if task_type else "overall_accuracy"
    if acc_col not in df.columns:
        raise ValueError(f"Column '{acc_col}' not found.")

    configs = sorted(df["config"].unique())
    cfg_palette = {c: _DEFAULT_PALETTE[i] for i, c in enumerate(configs)}

    agg = (
        df.groupby(["seed", "config"])[acc_col]
        .agg(mean_acc="mean", sd_acc="std")
        .reset_index()
        .dropna()
    )

    ncols = len(configs)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, cfg in zip(axes, configs):
        sub = agg[agg["config"] == cfg]
        ax.scatter(
            sub["mean_acc"], sub["sd_acc"],
            color=cfg_palette[cfg], s=55, alpha=0.75,
            edgecolors="white", linewidths=0.5,
        )
        ax.axhline(sub["sd_acc"].mean(), color=cfg_palette[cfg],
                   linestyle="--", linewidth=1.2, alpha=0.7, label="Mean SD")
        ax.set_title(_config_label(cfg), fontsize=13)
        ax.set_xlabel("Mean Accuracy Across Models", fontsize=12)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(linestyle="--", alpha=0.35)
        ax.legend(fontsize=10)

    axes[0].set_ylabel("SD of Accuracy Across Models", fontsize=12)
    fig.suptitle(
        "Seed Discriminability — Model Agreement Per Seed"
        + (f" — task: {task_type}" if task_type else ""),
        fontsize=14,
    )
    fig.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stats summary table figure
# ─────────────────────────────────────────────────────────────────────────────

def render_stats_table(
    stats_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (13, 4),
    title: str = "Seed Stability — Statistical Summary",
) -> None:
    """
    Render the ICC / Kruskal-Wallis / variance summary DataFrame as a
    publication-quality matplotlib table figure.

    Parameters
    ----------
    stats_df    : DataFrame from analyze_reseeding.build_stats_table().
                  Expected columns (any subset works):
                    model, config, mean_acc, std_acc, cv_pct,
                    kw_stat, kw_pvalue, kw_significant,
                    icc, icc_ci_low, icc_ci_high, n_seeds
    output_path : File path for the saved figure.
    """
    display_cols = {
        "model":           "Model",
        "config":          "Config",
        "n_seeds":         "Seeds",
        "mean_acc":        "Mean Acc",
        "std_acc":         "SD",
        "cv_pct":          "CV (%)",
        "kw_stat":         "KW stat",
        "kw_pvalue":       "KW p",
        "icc":             "ICC(2,1)",
        "icc_ci_low":      "ICC 95% CI",
    }
    present = [c for c in display_cols if c in stats_df.columns]
    sub = stats_df[present].copy()
    sub.rename(columns={c: display_cols[c] for c in present}, inplace=True)

    # Format numeric columns
    fmt_2f = ["Mean Acc", "SD", "ICC(2,1)", "ICC 95% CI"]
    fmt_1f = ["CV (%)", "KW stat"]
    fmt_4f = ["KW p"]
    for col in sub.columns:
        if col in fmt_2f:
            sub[col] = sub[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        elif col in fmt_1f:
            sub[col] = sub[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        elif col in fmt_4f:
            sub[col] = sub[col].apply(
                lambda x: (f"{x:.4f}" + (" *" if isinstance(x, float) and x < 0.05 else ""))
                if pd.notna(x) else "—"
            )

    # Merge ICC CI columns into one string if both exist
    if "icc_ci_low" in stats_df.columns and "icc_ci_high" in stats_df.columns:
        sub["ICC 95% CI"] = stats_df.apply(
            lambda r: f"[{r.icc_ci_low:.3f}, {r.icc_ci_high:.3f}]"
            if pd.notna(r.get("icc_ci_low")) else "—",
            axis=1,
        )

    nrows, ncols = sub.shape
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    tbl = ax.table(
        cellText=sub.values,
        colLabels=sub.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)

    # Style header row
    for j in range(ncols):
        tbl[0, j].set_facecolor("#263238")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Zebra stripes + highlight significant KW rows
    kw_col_idx = list(sub.columns).index("KW p") if "KW p" in sub.columns else None
    for i in range(1, nrows + 1):
        bg = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(ncols):
            tbl[i, j].set_facecolor(bg)
        if kw_col_idx is not None:
            val = sub.iloc[i - 1, kw_col_idx]
            if isinstance(val, str) and "*" in val:
                for j in range(ncols):
                    tbl[i, j].set_facecolor("#FFEBEE")

    ax.set_title(title, fontsize=14, pad=16, fontweight="bold")
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Depth accuracy box plot  (new)
# ─────────────────────────────────────────────────────────────────────────────

# Deep, saturated colors — one per model
_DEEP_COLORS = [
    "#1565C0",  # deep blue
    "#B71C1C",  # deep red
    "#2E7D32",  # deep green
    "#6A1B9A",  # deep purple
    "#E65100",  # deep orange
    "#00695C",  # deep teal
    "#AD1457",  # deep pink
    "#4E342E",  # deep brown
    "#37474F",  # dark blue-grey
]

_DEPTH_YLABEL: dict[str, str] = {
    "default_reasoning":  "Chain Length",
    "linear_inheritance": "Hierarchy Depth",
    "tree_inheritance":   "Graph Size (# Nodes)",
}


def _bin_label(depth_val: int, bin_step: int) -> str:
    """Map a depth integer to a bin label string."""
    if bin_step <= 1:
        return str(depth_val)
    lo = ((depth_val - 1) // bin_step) * bin_step + 1
    hi = lo + bin_step - 1
    return f"{lo}–{hi}"


def _bin_sort_key(depth_val: int, bin_step: int) -> int:
    """Numeric sort key for a binned depth value (= lower bound of its bin)."""
    if bin_step <= 1:
        return depth_val
    return ((depth_val - 1) // bin_step) * bin_step + 1


def plot_depth_accuracy_boxplot(
    df: pd.DataFrame,
    output_path: str,
    topology: str = "default_reasoning",
    difficulty: Optional[str] = None,
    bin_step: int = 2,
    font_size: int = 14,
    font_color: str = "#1a1a2e",
    font_family: str = "Times New Roman",
    figsize: tuple = (6, 4),
) -> None:
    """
    Horizontal box plot: x = accuracy, y = binned chain-length / node-count.

    Within each (model × seed × bin) group, sample accuracies are averaged
    before plotting, so each data point in a box represents one seed's mean
    accuracy for samples that fall in that depth bin.

    Parameters
    ----------
    df          : DataFrame from analyze_reseeding.build_depth_accuracy_df().
                  Required columns: model_id, seed, topology, difficulty,
                  depth_val, accuracy.
    output_path : Saved figure path (.pdf / .png / .svg).
    topology    : Filter to this topology.
    difficulty  : If None, pool all difficulties; otherwise filter.
    bin_step    : Group every bin_step consecutive depth values into one bin.
                  bin_step=2 → bins "1–2", "3–4", "5–6", …
                  bin_step=1 → each depth value is its own bin.
    font_size   : Base font size for axis labels and ticks.
    font_color  : Text colour applied to all labels, ticks, and title.
    font_family : Font family name (default "Times New Roman").
                  Falls back to "DejaVu Serif" if not installed.
    figsize     : (width, height) in inches.
    """
    # ── 1. Filter ────────────────────────────────────────────────────────────
    sub = df[df["topology"] == topology].copy()
    if difficulty:
        sub = sub[sub["difficulty"] == difficulty]
    if sub.empty:
        print(
            f"  [Skip] depth_accuracy_boxplot — no data for "
            f"topology={topology!r} difficulty={difficulty!r}"
        )
        return

    # ── 2. Short model names ─────────────────────────────────────────────────
    sub["model_name"] = sub["model_id"].apply(_short)

    # ── 3. Bin depth values ──────────────────────────────────────────────────
    sub["bin_label"]    = sub["depth_val"].apply(lambda v: _bin_label(v, bin_step))
    sub["_bin_sort_key"] = sub["depth_val"].apply(lambda v: _bin_sort_key(v, bin_step))

    # ── 4. Aggregate: mean accuracy per (model, seed, bin) ───────────────────
    agg = (
        sub.groupby(["model_name", "seed", "bin_label", "_bin_sort_key"])["accuracy"]
        .mean()
        .reset_index()
    )

    # ── 5. Ordered y-axis labels (ascending by bin lower bound) ─────────────
    bin_order = (
        agg[["bin_label", "_bin_sort_key"]]
        .drop_duplicates()
        .sort_values("_bin_sort_key")["bin_label"]
        .tolist()
    )

    models   = sorted(agg["model_name"].unique())
    n_models = len(models) # _DEFAULT_PALETTE
    # palette  = {m: _DEEP_COLORS[i % len(_DEEP_COLORS)] for i, m in enumerate(models)}
    palette  = {m: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)] for i, m in enumerate(models)}

    # ── 6. Font resolution ───────────────────────────────────────────────────
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    if font_family not in available_fonts:
        print(f"  [Warn] Font '{font_family}' not found; falling back to 'Times'.")
        font_family = "Times"

    # ── 7. Draw ──────────────────────────────────────────────────────────────
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
        # Taller figure when many bins and models (avoid overlap)
        n_bins = len(bin_order)
        auto_h = max(figsize[1], 0.2 * n_bins * max(n_models, 1) + .7)
        fig, ax = plt.subplots(figsize=(figsize[0], auto_h))

        bp_kw = dict(
            data=agg,
            x="accuracy",
            y="bin_label",
            order=bin_order,
            linewidth=1.5,
            flierprops=flier_kw,
            ax=ax,
        )

        if n_models > 1:
            sns.boxplot(
                **bp_kw,
                hue="model_name",
                hue_order=models,
                palette=palette,
                width=0.6,
            )
            legend = ax.get_legend()
            if legend:
                legend.set_title("Model", prop={"size": font_size - 1, "family": font_family})
                for text in legend.get_texts():
                    text.set_fontsize(font_size - 1)
                    text.set_color(font_color)
        else:
            sns.boxplot(**bp_kw, color=_DEEP_COLORS[0], width=0.45)

        # ── Axes formatting ─────────────────────────────────────────────────
        y_label = _DEPTH_YLABEL.get(topology, "Depth")
        bin_suffix = f"  (bin size = {bin_step})" if bin_step > 1 else ""
        ax.set_xlabel("Accuracy", fontsize=font_size + 1)
        ax.set_ylabel(y_label + bin_suffix, fontsize=font_size + 1)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xlim(-0.02, 1.02)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Vertical x-grid only (keeps the plot uncluttered)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45, color="#aaaaaa")
        ax.set_axisbelow(True)

        # ── Title ────────────────────────────────────────────────────────────
        topo_label = {
            "default_reasoning":  "Default Reasoning",
            "linear_inheritance": "Linear Inheritance",
            "tree_inheritance":   "Tree Inheritance",
        }.get(topology, topology.replace("_", " ").title())
        diff_str = f" · {difficulty.capitalize()}" if difficulty else ""
        ax.set_title(
            f"Accuracy by {y_label}  ·  {topo_label}{diff_str}",
            fontsize=font_size + 2, pad=14, fontweight="bold",
        )

        fig.tight_layout()

    _save(fig, output_path)
