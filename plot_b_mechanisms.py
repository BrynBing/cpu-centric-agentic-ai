
"""Detailed Fixed-B vs Dynamic-B scheduling visualizations."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "fixed-b": "#64748b",
    "dynamic-b": "#0f766e",
}


def save(fig, output_dir, filename):
    fig.savefig(
        output_dir / filename,
        bbox_inches="tight",
        dpi=180,
        facecolor="white",
    )
    plt.close(fig)
    print("Saved:", filename)


def load_runs(root):
    runs = {}

    for folder in sorted(root.glob("[0-9][0-9][0-9]_*")):
        if not folder.is_dir():
            continue

        try:
            repeat, mode = folder.name.split("_", 1)

            if mode not in COLORS:
                continue

            runs[(int(repeat), mode)] = {
                name: pd.read_csv(folder / f"{name}.csv")
                for name in (
                    "requests",
                    "cohorts",
                    "events",
                    "trace",
                )
            }

        except (
            ValueError,
            FileNotFoundError,
            pd.errors.EmptyDataError,
        ) as exc:
            print("Skipping incomplete run:", folder, exc)

    if not runs:
        raise ValueError("No completed B experiments found.")

    return runs


def combine_mode(runs, table):
    result = {}

    for mode in COLORS:
        frames = []

        for (repeat, current_mode), data in runs.items():
            if current_mode == mode:
                frames.append(
                    data[table].assign(repeat=repeat)
                )

        if frames:
            result[mode] = pd.concat(
                frames,
                ignore_index=True,
            )

    return result


def plot_batch_behavior(runs, output_dir):
    cohorts = combine_mode(runs, "cohorts")

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5),
        constrained_layout=True,
    )

    for mode, frame in cohorts.items():
        counts = (
            frame["actual"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        axes[0].plot(
            counts.index,
            counts.values,
            "-o",
            label=mode,
            color=COLORS[mode],
        )

        first_repeat = frame["repeat"].min()

        selected = frame[
            frame["repeat"] == first_repeat
        ].sort_values("cohort_id")

        axes[1].step(
            selected["cohort_id"],
            selected["target"],
            where="mid",
            label=mode,
            color=COLORS[mode],
        )

    axes[0].set_title("Actual cohort-size distribution")
    axes[0].set_xlabel("Actual cohort size")
    axes[0].set_ylabel("Number of cohorts")

    axes[1].set_title("Target B over scheduling decisions")
    axes[1].set_xlabel("Cohort ID")
    axes[1].set_ylabel("Target batch size")

    for ax in axes:
        ax.legend()

    save(fig, output_dir, "01_batch_behavior.png")


def plot_decision_reasons(runs, output_dir):
    reasons = []

    for (repeat, mode), data in runs.items():
        if mode != "dynamic-b":
            continue

        events = data["events"]
        selected = events[
            events["event"] == "batch_decision"
        ]

        for value in selected["detail"]:
            try:
                detail = json.loads(value)
                reasons.append(
                    detail.get("reason", "unknown")
                )
            except (TypeError, ValueError):
                reasons.append("unparseable")

    counts = pd.Series(reasons, dtype=str).value_counts()

    fig, ax = plt.subplots(figsize=(9, 4.5))

    if not counts.empty:
        ax.barh(
            counts.index[::-1],
            counts.values[::-1],
            color=COLORS["dynamic-b"],
        )

    ax.set_title("Dynamic-B decision reasons")
    ax.set_xlabel("Number of decisions")

    save(
        fig,
        output_dir,
        "02_dynamic_decision_reasons.png",
    )


def plot_latency_cdf(runs, output_dir):
    requests = combine_mode(runs, "requests")

    metrics = [
        ("latency", "End-to-end latency"),
        ("service_latency", "Service latency"),
        ("queue_wait", "Queue waiting time"),
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4.5),
        constrained_layout=True,
    )

    for ax, (metric, title) in zip(axes, metrics):
        for mode, frame in requests.items():
            successful = frame[
                frame["success"] == True
            ]

            values = np.sort(
                pd.to_numeric(
                    successful[metric],
                    errors="coerce",
                ).dropna().to_numpy()
            )

            if len(values) == 0:
                continue

            cdf = np.arange(
                1, len(values) + 1
            ) / len(values)

            ax.step(
                values,
                cdf,
                where="post",
                label=mode,
                color=COLORS[mode],
            )

        ax.set_title(title)
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Empirical CDF")

    axes[0].legend()

    fig.suptitle(
        "Request latency distributions across repeats"
    )

    save(
        fig,
        output_dir,
        "03_request_latency_cdf.png",
    )


def plot_latency_breakdown(runs, output_dir):
    records = []

    for (repeat, mode), data in runs.items():
        frame = data["requests"]

        phases = {
            "Queue": frame["queue_wait"],
            "Preparation": frame["ready"] - frame["start"],
            "Pre-inference gap":
                frame["inference_start"] - frame["ready"],
            "Inference client":
                frame["finish"] - frame["inference_start"],
            "Post-inference":
                frame["exit"] - frame["finish"],
        }

        for phase, values in phases.items():
            numbers = (
                pd.to_numeric(values, errors="coerce")
                .dropna()
                .clip(lower=0)
            )

            if numbers.empty:
                continue

            records.append({
                "repeat": repeat,
                "mode": mode,
                "phase": phase,
                "mean": numbers.mean(),
                "median": numbers.median(),
            })

    df = pd.DataFrame(records)

    phases = [
        "Queue",
        "Preparation",
        "Pre-inference gap",
        "Inference client",
        "Post-inference",
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.5),
        constrained_layout=True,
    )

    modes = ["fixed-b", "dynamic-b"]

    for ax, statistic in zip(
        axes, ["mean", "median"]
    ):
        bottom = np.zeros(len(modes))
        positions = np.arange(len(modes))

        for phase in phases:
            values = []

            for mode in modes:
                subset = df[
                    (df["mode"] == mode)
                    & (df["phase"] == phase)
                ]

                values.append(
                    subset[statistic].mean()
                    if not subset.empty
                    else 0
                )

            values = np.asarray(values)

            ax.bar(
                positions,
                values,
                bottom=bottom,
                label=phase,
            )
            bottom += values

        ax.set_xticks(
            positions,
            ["Fixed-B", "Dynamic-B"],
        )
        ax.set_ylabel("Seconds")
        ax.set_title(
            f"{statistic.title()} phase durations"
        )

    axes[1].legend(
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
    )

    fig.suptitle(
        "Phase durations; medians are calculated separately"
    )

    save(
        fig,
        output_dir,
        "04_stage_duration_breakdown.png",
    )


def plot_cumulative_completion(runs, output_dir):
    repeats = sorted({
        repeat for repeat, mode in runs
    })

    count = min(3, len(repeats))

    fig, axes = plt.subplots(
        1,
        count,
        figsize=(5 * count, 4.5),
        constrained_layout=True,
        squeeze=False,
    )

    for ax, repeat in zip(
        axes.flat, repeats[:count]
    ):
        for mode in COLORS:
            if (repeat, mode) not in runs:
                continue

            frame = runs[
                (repeat, mode)
            ]["requests"]

            completed = np.sort(
                frame.loc[
                    frame["success"] == True,
                    "exit",
                ].dropna().to_numpy()
            )

            if len(completed) == 0:
                continue

            ax.step(
                np.r_[0, completed],
                np.r_[0, np.arange(
                    1, len(completed) + 1
                )],
                where="post",
                label=mode,
                color=COLORS[mode],
            )

        ax.set_title(f"Repeat {repeat}")
        ax.set_xlabel("Elapsed seconds")
        ax.set_ylabel("Completed requests")

    axes.flat[0].legend()

    save(
        fig,
        output_dir,
        "05_cumulative_completions.png",
    )


def plot_request_timeline(runs, output_dir):
    repeat = min(
        repeat for repeat, mode in runs
    )

    fig, axes = plt.subplots(
        2, 1,
        figsize=(13, 7),
        sharex=True,
        constrained_layout=True,
    )

    for ax, mode in zip(axes, COLORS):
        if (repeat, mode) not in runs:
            continue

        frame = (
            runs[(repeat, mode)]["requests"]
            .sort_values("start")
            .head(24)
        )

        for position, (_, row) in enumerate(
            frame.iterrows()
        ):
            intervals = [
                (
                    row["start"],
                    row["ready"],
                    "#0d9488",
                ),
                (
                    row["inference_start"],
                    row["finish"],
                    "#d97706",
                ),
            ]

            for start, end, color in intervals:
                if pd.isna(start) or pd.isna(end):
                    continue

                if end < start:
                    continue

                ax.barh(
                    position,
                    end - start,
                    left=start,
                    height=0.7,
                    color=color,
                )

        ax.set_title(
            f"{mode} — Repeat {repeat}"
        )
        ax.set_ylabel("Request execution order")
        ax.invert_yaxis()

    axes[-1].set_xlabel(
        "Elapsed seconds (client-observed)"
    )

    fig.suptitle(
        "Green: preparation | Orange: inference client"
    )

    save(
        fig,
        output_dir,
        "06_request_stage_timeline.png",
    )


def plot_cpu_trace(runs, output_dir):
    repeat = min(
        repeat for repeat, mode in runs
    )

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 6),
        sharex=True,
        constrained_layout=True,
    )

    for mode in COLORS:
        if (repeat, mode) not in runs:
            continue

        frame = runs[(repeat, mode)]["trace"]

        axes[0].plot(
            frame["time"],
            frame["cpu_util"] * 100,
            label=mode,
            color=COLORS[mode],
            alpha=0.8,
        )

        axes[1].step(
            frame["time"],
            frame["active"],
            where="post",
            label=mode,
            color=COLORS[mode],
        )

    axes[0].set_title(
        f"CPU and active worker trace — Repeat {repeat}"
    )
    axes[0].set_ylabel("CPU utilization (%)")

    axes[1].set_ylabel("Active workers")
    axes[1].set_xlabel("Elapsed seconds")

    axes[0].legend()

    save(
        fig,
        output_dir,
        "07_cpu_active_trace.png",
    )


def plot_paired_comparison(runs, output_dir):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 4.5),
        constrained_layout=True,
    )

    metrics = [
        ("makespan", "Makespan (lower is better)"),
        ("goodput", "Goodput (higher is better)"),
    ]

    repeats = sorted({
        repeat for repeat, mode in runs
    })

    for ax, (metric, title) in zip(
        axes, metrics
    ):
        for repeat in repeats:
            if not all(
                (repeat, mode) in runs
                for mode in COLORS
            ):
                continue

            values = []

            for mode in COLORS:
                requests = runs[
                    (repeat, mode)
                ]["requests"]

                elapsed = float(
                    requests["exit"].max()
                )

                successes = int(
                    (requests["success"] == True).sum()
                )

                if metric == "makespan":
                    value = elapsed
                else:
                    value = successes / elapsed

                values.append(value)

            ax.plot(
                [0, 1],
                values,
                "-o",
                label=f"Repeat {repeat}",
            )

        ax.set_xticks(
            [0, 1],
            ["Fixed-B", "Dynamic-B"],
        )
        ax.set_title(title)
        ax.legend()

    save(
        fig,
        output_dir,
        "08_paired_runs.png",
    )


def write_findings(runs, output_dir):
    lines = [
        "B scheduling diagnostics",
        "========================",
        "",
    ]

    for mode in COLORS:
        data = [
            item for (repeat, current_mode), item
            in runs.items()
            if current_mode == mode
        ]

        if not data:
            continue

        cohorts = pd.concat(
            [item["cohorts"] for item in data]
        )

        requests = pd.concat(
            [item["requests"] for item in data]
        )

        lines.extend([
            f"{mode}:",
            f"Runs: {len(data)}",
            f"Requests: {len(requests)}",
            f"Cohorts: {len(cohorts)}",
            "Actual cohort size distribution:",
            str(
                cohorts["actual"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            f"Median queue wait: "
            f"{requests['queue_wait'].median():.3f}s",
            f"Median service latency: "
            f"{requests['service_latency'].median():.3f}s",
            "",
        ])

    lines.extend([
        "Important interpretation notes:",
        "",
        "Cohort size represents client-side request grouping.",
        "It is not necessarily the GPU inference batch size.",
        "",
        "Inference intervals are measured on the client.",
        "They do not establish GPU kernel execution overlap.",
        "",
        "Missing GPU metrics must not be treated as zero.",
        "",
        "Pooled request CDFs are descriptive.",
        "Paired experimental runs should be used for",
        "independent trial-level performance comparisons.",
        "",
        "If B never changes, a measured performance difference",
        "cannot be attributed to adaptive batch-size selection.",
    ])

    (output_dir / "README_findings.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    root = args.results_dir.resolve()

    runs = load_runs(root)

    output_dir = root / "figures_mechanisms"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "figure.facecolor": "white",
    })

    plot_batch_behavior(runs, output_dir)
    plot_decision_reasons(runs, output_dir)
    plot_latency_cdf(runs, output_dir)
    plot_latency_breakdown(runs, output_dir)
    plot_cumulative_completion(runs, output_dir)
    plot_request_timeline(runs, output_dir)
    plot_cpu_trace(runs, output_dir)
    plot_paired_comparison(runs, output_dir)

    write_findings(runs, output_dir)

    print("\nAll figures saved to:", output_dir)


if __name__ == "__main__":
    main()
