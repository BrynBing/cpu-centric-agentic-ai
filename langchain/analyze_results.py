import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)


# --------------------------------------------------
# 1. Load experiment results
# --------------------------------------------------

experiments = {}

static_files = glob.glob(
    os.path.join(RESULT_DIR, "static_n*.csv")
)

for path in static_files:

    match = re.search(r"static_n(\d+)", path)

    if match:
        n = int(match.group(1))
        name = f"Static N={n}"

        experiments[name] = pd.read_csv(path)


dynamic_path = os.path.join(
    RESULT_DIR,
    "dynamic_jobs.csv"
)

if os.path.exists(dynamic_path):
    experiments["Dynamic N"] = pd.read_csv(dynamic_path)


# --------------------------------------------------
# 2. Calculate metrics
# --------------------------------------------------

summary = []

for name, df in experiments.items():

    latency = df["latency"].to_numpy()

    p50 = np.percentile(latency, 50)
    p90 = np.percentile(latency, 90)

    makespan = df["finish_time"].max()

    throughput = len(df) / makespan

    mean_latency = latency.mean()

    summary.append({
        "scheduler": name,
        "requests": len(df),
        "mean_latency": mean_latency,
        "p50": p50,
        "p90": p90,
        "makespan": makespan,
        "throughput": throughput
    })


summary_df = pd.DataFrame(summary)

summary_df = summary_df.sort_values(
    by="scheduler"
)

summary_path = os.path.join(
    RESULT_DIR,
    "summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\nExperiment Summary")
print(summary_df.to_string(index=False))


# --------------------------------------------------
# 3. P50 / P90 latency comparison
# --------------------------------------------------

if not summary_df.empty:

    labels = summary_df["scheduler"]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        summary_df["p50"],
        width,
        label="P50"
    )

    ax.bar(
        x + width / 2,
        summary_df["p90"],
        width,
        label="P90"
    )

    ax.set_ylabel("Latency (s)")
    ax.set_title("P50 and P90 Latency Comparison")

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULT_DIR,
            "latency_comparison.png"
        ),
        dpi=300
    )

    plt.close(fig)


# --------------------------------------------------
# 4. Completion percentile curves
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

for name, df in experiments.items():

    completion = np.sort(
        df["finish_time"].to_numpy()
    )

    percentile = (
        np.arange(1, len(completion) + 1)
        / len(completion)
        * 100
    )

    ax.plot(
        completion,
        percentile,
        label=name
    )

ax.set_xlabel("Completion time (s)")
ax.set_ylabel("Completed requests (%)")

ax.set_title(
    "Request Completion Percentile"
)

ax.set_ylim(0, 100)

ax.legend()

fig.tight_layout()

fig.savefig(
    os.path.join(
        RESULT_DIR,
        "completion_curve.png"
    ),
    dpi=300
)

plt.close(fig)


# --------------------------------------------------
# 5. Dynamic scheduler traces
# --------------------------------------------------

trace_path = os.path.join(
    RESULT_DIR,
    "dynamic_trace.csv"
)

if os.path.exists(trace_path):

    trace = pd.read_csv(trace_path)

    # N(t)
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        trace["time"],
        trace["n_cap"]
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Concurrency cap N")

    ax.set_title(
        "Dynamic Concurrency Cap N(t)"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULT_DIR,
            "dynamic_n.png"
        ),
        dpi=300
    )

    plt.close(fig)


    # CPU utilisation
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        trace["time"],
        trace["cpu_util"]
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CPU utilisation (%)")

    ax.set_title(
        "CPU Utilisation Over Time"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULT_DIR,
            "dynamic_cpu.png"
        ),
        dpi=300
    )

    plt.close(fig)


print("\nGenerated figures:")
print(" - latency_comparison.png")
print(" - completion_curve.png")
print(" - dynamic_n.png")
print(" - dynamic_cpu.png")