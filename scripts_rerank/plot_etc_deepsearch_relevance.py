import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.lines import Line2D


RUN_SOURCES = {
    "20b_fixed": "/u6/s8sharif/BrowseComp-Plus/runs_fixed_passage_length/Qwen3-Embedding-8B/gpt-oss-20b/aggregated_results.tsv",
    "20b_rel": "/u501/hoyarhos/BrowseComp-Plus/relevance_runs/Qwen3-Embedding-8B/gpt-oss-20b/aggregated_results.tsv",
    "120b_fixed": "/u6/s8sharif/BrowseComp-Plus/runs_fixed_passage_length/Qwen3-Embedding-8B/gpt-oss-120b/aggregated_results.tsv",
    "120b_rel": "/u501/hoyarhos/BrowseComp-Plus/relevance_runs/Qwen3-Embedding-8B/gpt-oss-120b/aggregated_results.tsv",
}


BASE_RUN_GROUPS = [
    {
        "group_id": "group1",
        "group_name": "gpt-oss-20b-low (d=10)",
        "color": "blue",
        "runs": [
            {
                "source_key": "20b_fixed",
                "run_dir": "rerank_none_search_rf_low_k_5_doc_length_512_job_1417959",
                "point_label": "(1a) None",
                "rel_key": None,
            },
            {
                "source_key": "20b_fixed",
                "run_dir": "rerank_rf_low_k_10_search_rf_low_k_5_doc_length_512_job_1416621",
                "point_label": "(1b) Rerank",
                "rel_key": None,
            },
            {
                "source_key": "20b_rel",
                "run_dir": "relevance_rf_low_k_10_search_rf_low_k_5_doc_length_512_job_1417895_sub_only",
                "point_label": "(1c) Rel (sub-q)",
                "rel_key": "sub_only",
            },
            {
                "source_key": "20b_rel",
                "run_dir": "relevance_rf_low_k_10_search_rf_low_k_5_doc_length_512_job_1417895_query_sub",
                "point_label": "(1d) Rel (sub-q + q)",
                "rel_key": "query_sub",
            },
            {
                "source_key": "20b_rel",
                "run_dir": "relevance_rf_low_k_10_search_rf_low_k_5_doc_length_512_job_1417895_sub_reason",
                "point_label": "(1e) Rel (sub-q + r)",
                "rel_key": "sub_reason",
            },
            {
                "source_key": "20b_rel",
                "run_dir": "relevance_rf_low_k_10_search_rf_low_k_5_doc_length_512_job_1417895_all_three",
                "point_label": "(1f) Rel (sub-q + q + r)",
                "rel_key": "all_three",
            },
        ],
    },
    {
        "group_id": "group2",
        "group_name": "gpt-oss-120b-high (d=50)",
        "color": "red",
        "runs": [
            {
                "source_key": "120b_fixed",
                "run_dir": "rerank_none_search_rf_high_k_5_doc_length_512_job_1418391",
                "point_label": "(2a) None",
                "rel_key": None,
            },
            {
                "source_key": "120b_fixed",
                "run_dir": "rerank_rf_low_k_50_search_rf_high_k_5_doc_length_512_job_1418392",
                "point_label": "(2b) Rerank",
                "rel_key": None,
            },
            {
                "source_key": "120b_rel",
                "run_dir": "relevance_rf_low_k_50_search_rf_high_k_5_doc_length_512_job_1417893_sub_only",
                "point_label": "(2c) Rel (sub-q)",
                "rel_key": "sub_only",
            },
            {
                "source_key": "120b_rel",
                "run_dir": "relevance_rf_low_k_50_search_rf_high_k_5_doc_length_512_job_1417893_query_sub",
                "point_label": "(2d) Rel (sub-q + q)",
                "rel_key": "query_sub",
            },
            {
                "source_key": "120b_rel",
                "run_dir": "relevance_rf_low_k_50_search_rf_high_k_5_doc_length_512_sub_reason",
                "point_label": "(2e) Rel (sub-q + r)",
                "rel_key": "sub_reason",
            },
            {
                "source_key": "120b_rel",
                "run_dir": "relevance_rf_low_k_50_search_rf_high_k_5_doc_length_512_all_three",
                "point_label": "(2f) Rel (sub-q + q + r)",
                "rel_key": "all_three",
            },
        ],
    },
]


def load_tsv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required TSV not found: {path}")
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_metric_name(metric, columns):
    if metric.lower() in ["accuracy", "acc"]:
        return "Acc 120b Avg" if "Acc 120b Avg" in columns else metric

    if metric in columns:
        return metric

    lower_map = {c.lower(): c for c in columns}
    return lower_map.get(metric.lower(), metric)


def to_float(val):
    parsed = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else None


def choose_nice_step(span, target_intervals=6):
    """Pick a human-friendly major tick step for a given data span."""
    if span <= 0:
        return 1.0
    candidates = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    raw = span / max(target_intervals, 1)
    for c in candidates:
        if c >= raw:
            return float(c)
    return float(candidates[-1])


def compute_square_limits(x_values, y_values, x_step, y_step, pad_ticks=1):
    """
    Build per-panel limits that:
    1) snap to tick grid,
    2) include padding.
    """
    x_min_data, x_max_data = min(x_values), max(x_values)
    y_min_data, y_max_data = min(y_values), max(y_values)

    x_min_i = int((x_min_data // x_step) - pad_ticks)
    x_max_i = int(-(-x_max_data // x_step) + pad_ticks)  # ceil division for floats
    y_min_i = int((y_min_data // y_step) - pad_ticks)
    y_max_i = int(-(-y_max_data // y_step) + pad_ticks)

    return {
        "x_min": x_min_i * x_step,
        "x_max": x_max_i * x_step,
        "y_min": y_min_i * y_step,
        "y_max": y_max_i * y_step,
    }


def build_run_groups(rel_keep):
    groups = []
    for group in BASE_RUN_GROUPS:
        new_group = {k: v for k, v in group.items() if k != "runs"}
        new_runs = []
        for run in group["runs"]:
            rel_key = run.get("rel_key")
            if rel_key is None:
                new_runs.append(dict(run))
                continue
            if rel_keep == "all" or rel_key == rel_keep:
                run_copy = dict(run)
                if rel_keep != "all":
                    run_copy["point_label"] = "Relevance"
                new_runs.append(run_copy)
        new_group["runs"] = new_runs
        groups.append(new_group)
    return groups


def build_plot_rows(metric, rel_keep="all_three"):
    loaded = {key: load_tsv(path) for key, path in RUN_SOURCES.items()}

    all_columns = set()
    for df in loaded.values():
        all_columns.update(df.columns.tolist())
    metric_col = normalize_metric_name(metric, all_columns)
    if metric_col not in all_columns:
        raise ValueError(f"Metric '{metric}' not found in TSVs.")

    rows = []
    for group in build_run_groups(rel_keep):
        for idx, run in enumerate(group["runs"]):
            df = loaded[run["source_key"]]
            matched = df[df["Run Directory"] == run["run_dir"]]
            if matched.empty:
                raise ValueError(f"Run directory '{run['run_dir']}' not found in {RUN_SOURCES[run['source_key']]}")

            row = matched.iloc[0]
            metric_val = to_float(row.get(metric_col))
            if metric_val is None:
                raise ValueError(f"Metric '{metric_col}' missing/non-numeric for run '{run['run_dir']}'.")

            input_tokens = to_float(row.get("input_tokens")) or 0.0
            cached_tokens = to_float(row.get("cached_tokens")) or 0.0
            output_tokens = to_float(row.get("output_tokens")) or 0.0
            ih_input_tokens = to_float(row.get("ih_input_tokens")) or 0.0
            ih_cached_tokens = to_float(row.get("ih_cached_tokens")) or 0.0
            ih_output_tokens = to_float(row.get("ih_output_tokens")) or 0.0

            rows.append(
                {
                    "group_id": group["group_id"],
                    "group_name": group["group_name"],
                    "color": group["color"],
                    "order": idx,
                    "point_label": run["point_label"],
                    "run_dir": run["run_dir"],
                    "metric": metric_val,
                    "input_total": input_tokens + ih_input_tokens,
                    "cached_total": cached_tokens + ih_cached_tokens,
                    "output_total": output_tokens + ih_output_tokens,
                }
            )

    return pd.DataFrame(rows), metric_col


def plot_etc(metric="accuracy", rel_keep="all_three"):
    alphas = [0.1, 0.3, 0.5]
    betas = [3, 5, 7]

    plot_df, metric_col = build_plot_rows(metric, rel_keep=rel_keep)
    if plot_df.empty:
        print("No rows available to plot.")
        return

    per_ab_df = {}

    for alpha in alphas:
        for beta in betas:
            local = plot_df.copy()
            local["etc"] = (
                (local["input_total"] - local["cached_total"])
                + alpha * local["cached_total"]
                + beta * local["output_total"]
            )

            local["etc_billions"] = local["etc"] / 1e9
            local["etc_abs_billions"] = local["etc"] / 1e9
            # Use G1 None as global accuracy baseline for both groups.
            g1_rows = local[local["group_id"] == "group1"].sort_values("order")
            if g1_rows.empty:
                raise ValueError("Could not find G1 rows for global baseline.")
            g1_baseline_metric = g1_rows["metric"].iloc[0]
            local["metric_improvement_pct"] = (
                (local["metric"] - g1_baseline_metric) / g1_baseline_metric * 100.0
            )

            per_ab_df[(alpha, beta)] = local

    # Unified plot in billions on x-axis.
    x_step = 0.2
    y_step = 20.0

    global_xs = []
    global_ys = []
    for df_ab in per_ab_df.values():
        global_xs.extend(df_ab["etc_abs_billions"].tolist())
        global_ys.extend(df_ab["metric_improvement_pct"].tolist())
    global_limits = compute_square_limits(global_xs, global_ys, x_step, y_step, pad_ticks=1)

    for (alpha, beta), local in per_ab_df.items():
        fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.6))
        group_colors = {"group1": "#1f77b4", "group2": "#d62728"}
        marker_cycle = ["o", "s", "^", "D", "P", "X"]

        for group_id in ["group1", "group2"]:
            g = local[local["group_id"] == group_id].sort_values("order")
            g = g.sort_values("order")

            for i, (_, row) in enumerate(g.iterrows()):
                ax.scatter(
                    row["etc_abs_billions"],
                    row["metric_improvement_pct"],
                    color=group_colors[group_id],
                    s=210,
                    marker=marker_cycle[i % len(marker_cycle)],
                    edgecolors="black",
                    linewidths=1.2,
                    zorder=2,
                )

        ax.grid(True, linestyle="--", alpha=0.28, linewidth=0.8)
        ax.tick_params(axis="both", which="major", labelsize=13)
        ax.set_xlim(global_limits["x_min"], global_limits["x_max"])
        ax.set_ylim(global_limits["y_min"], global_limits["y_max"])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        # Make major grid cells square by matching box aspect to interval counts.
        x_intervals = max((global_limits["x_max"] - global_limits["x_min"]) / x_step, 1e-9)
        y_intervals = max((global_limits["y_max"] - global_limits["y_min"]) / y_step, 1e-9)
        ax.set_box_aspect(y_intervals / x_intervals)
        ax.set_ylabel("Accuracy Improvement (%)", fontsize=16, fontweight="bold")
        ax.set_xlabel("ETC [Billion tokens]", fontsize=16, fontweight="bold")

        semantic_labels = []
        g1 = local[local["group_id"] == "group1"].sort_values("order")
        for _, row in g1.iterrows():
            parts = str(row["point_label"]).split(") ", 1)
            semantic_labels.append(parts[1] if len(parts) == 2 else str(row["point_label"]))

        shape_handles = [
            Line2D(
                [0], [0],
                marker=marker_cycle[i],
                color="w",
                markerfacecolor="#7a7a7a",
                markeredgecolor="#4f4f4f",
                markersize=10,
                label=semantic_labels[i],
            )
            for i in range(len(semantic_labels))
        ]
        color_handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=group_colors["group1"],
                markeredgecolor="white",
                markersize=10,
                label="oss-20b-low",
            ),
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=group_colors["group2"],
                markeredgecolor="white",
                markersize=10,
                label="oss-120b-high",
            ),
        ]

        combined_handles = color_handles + shape_handles
        ax.legend(
            handles=combined_handles,
            loc="lower right",
            bbox_to_anchor=(0.995, 0.02),
            framealpha=0.9,
            prop={"size": 12, "weight": "bold"},
        )
        plt.tight_layout()
        clean_metric = metric_col.replace("@", "_").replace(" ", "_")
        output_filename = f"etc_relevance_groups_{clean_metric}_a{alpha}_b{beta}.pdf"
        plt.savefig(output_filename, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"Saved: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="accuracy")
    parser.add_argument(
        "--rel-keep",
        type=str,
        default="all_three",
        choices=["all", "sub_only", "query_sub", "sub_reason", "all_three"],
        help="Which relevance variant to keep. Default keeps only sub-q+q+r and labels it as Relevance.",
    )
    args = parser.parse_args()
    plot_etc(args.metric, rel_keep=args.rel_keep)
