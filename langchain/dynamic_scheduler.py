import argparse
import csv
import os
import subprocess
import time

import psutil


# ============================================================
# 1. Command-line arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Dynamic concurrency scheduler for LangChain workload"
)

parser.add_argument(
    "--jobs",
    type=int,
    default=128,
    help="Total number of requests/jobs"
)

parser.add_argument(
    "--initial-n",
    type=int,
    default=16,
    help="Initial concurrency cap"
)

parser.add_argument(
    "--min-n",
    type=int,
    default=4,
    help="Minimum concurrency cap"
)

parser.add_argument(
    "--max-n",
    type=int,
    default=64,
    help="Maximum concurrency cap"
)

parser.add_argument(
    "--step",
    type=int,
    default=4,
    help="Amount by which N changes"
)

parser.add_argument(
    "--low-cpu",
    type=float,
    default=65.0,
    help="Increase N when CPU utilisation is below this value"
)

parser.add_argument(
    "--high-cpu",
    type=float,
    default=85.0,
    help="Decrease N when CPU utilisation is above this value"
)

parser.add_argument(
    "--interval",
    type=float,
    default=1.0,
    help="Scheduler monitoring interval in seconds"
)

parser.add_argument(
    "--output-dir",
    type=str,
    default="results",
    help="Directory for CSV results"
)

args = parser.parse_args()


# ============================================================
# 2. Experiment configuration
# ============================================================

TOTAL_JOBS = args.jobs

N_MIN = args.min_n
N_MAX = args.max_n
N = args.initial_n

STEP = args.step

LOW_CPU = args.low_cpu
HIGH_CPU = args.high_cpu

INTERVAL = args.interval

OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRACE_FILE = os.path.join(
    OUTPUT_DIR,
    "dynamic_trace.csv"
)

JOB_FILE = os.path.join(
    OUTPUT_DIR,
    "dynamic_jobs.csv"
)


# ============================================================
# 3. Scheduler state
# ============================================================

pending_jobs = list(
    range(1, TOTAL_JOBS + 1)
)

running = {}

job_results = {}

completed = 0

experiment_start = time.time()


# ============================================================
# 4. Start CSV trace file
# ============================================================

trace_file = open(
    TRACE_FILE,
    "w",
    newline=""
)

trace_writer = csv.writer(trace_file)

trace_writer.writerow([
    "time",
    "cpu_util",
    "n_cap",
    "running",
    "pending",
    "completed"
])


# ============================================================
# 5. Function for launching one workload
# ============================================================

def launch_job(job_id):

    launch_time = (
        time.time()
        - experiment_start
    )

    print(
        f"[LAUNCH] Job {job_id} "
        f"at {launch_time:.2f}s"
    )

    process = subprocess.Popen(
        [
            "python",
            "orchestrator.py",
            "--skip-web-search",
            "--job-id",
            str(job_id)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    job_results[job_id] = {
        "launch_time": launch_time,
        "finish_time": None,
        "latency": None,
        "return_code": None
    }

    return process


# ============================================================
# 6. Main dynamic scheduling loop
# ============================================================

print("=" * 60)
print("Dynamic N Scheduler")
print("=" * 60)

print(f"Total jobs : {TOTAL_JOBS}")
print(f"Initial N  : {N}")
print(f"N range    : {N_MIN} - {N_MAX}")
print(f"CPU range  : {LOW_CPU}% - {HIGH_CPU}%")
print(f"Step       : {STEP}")
print("=" * 60)


try:

    while pending_jobs or running:

        # ----------------------------------------------------
        # A. Check completed jobs
        # ----------------------------------------------------

        finished_jobs = []

        for job_id, process in running.items():

            return_code = process.poll()

            if return_code is not None:

                finished_jobs.append(
                    (job_id, return_code)
                )


        for job_id, return_code in finished_jobs:

            finish_time = (
                time.time()
                - experiment_start
            )

            job_results[job_id][
                "finish_time"
            ] = finish_time

            # All requests are treated as arriving at t = 0.
            # Therefore latency includes scheduler queueing time.
            job_results[job_id][
                "latency"
            ] = finish_time

            job_results[job_id][
                "return_code"
            ] = return_code

            del running[job_id]

            completed += 1

            print(
                f"[DONE] Job {job_id} "
                f"at {finish_time:.2f}s "
                f"(code={return_code})"
            )


        # ----------------------------------------------------
        # B. Measure CPU utilisation
        # ----------------------------------------------------

        cpu_util = psutil.cpu_percent(
            interval=INTERVAL
        )


        # ----------------------------------------------------
        # C. Dynamically adjust N
        # ----------------------------------------------------

        old_N = N

        if cpu_util > HIGH_CPU:

            N = max(
                N_MIN,
                N - STEP
            )

        elif cpu_util < LOW_CPU:

            N = min(
                N_MAX,
                N + STEP
            )

        # Between LOW_CPU and HIGH_CPU:
        # keep N unchanged.


        if N != old_N:

            print(
                f"[ADJUST] CPU={cpu_util:.1f}% "
                f"N: {old_N} -> {N}"
            )


        # ----------------------------------------------------
        # D. Admit new jobs
        # ----------------------------------------------------

        while (
            pending_jobs
            and len(running) < N
        ):

            job_id = pending_jobs.pop(0)

            running[job_id] = launch_job(
                job_id
            )


        # ----------------------------------------------------
        # E. Record scheduler state
        # ----------------------------------------------------

        current_time = (
            time.time()
            - experiment_start
        )

        trace_writer.writerow([
            current_time,
            cpu_util,
            N,
            len(running),
            len(pending_jobs),
            completed
        ])

        trace_file.flush()


        # ----------------------------------------------------
        # F. Print status
        # ----------------------------------------------------

        print(
            f"[STATUS] "
            f"Time={current_time:.1f}s | "
            f"CPU={cpu_util:.1f}% | "
            f"N={N} | "
            f"Running={len(running)} | "
            f"Pending={len(pending_jobs)} | "
            f"Completed={completed}"
        )


finally:

    trace_file.close()


# ============================================================
# 7. Save per-job results
# ============================================================

with open(
    JOB_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "job_id",
        "launch_time",
        "finish_time",
        "latency",
        "return_code"
    ])

    for job_id in sorted(job_results):

        result = job_results[job_id]

        writer.writerow([
            job_id,
            result["launch_time"],
            result["finish_time"],
            result["latency"],
            result["return_code"]
        ])


# ============================================================
# 8. Final summary
# ============================================================

total_time = (
    time.time()
    - experiment_start
)

successful = sum(
    1
    for result in job_results.values()
    if result["return_code"] == 0
)

failed = (
    len(job_results)
    - successful
)


print("\n" + "=" * 60)
print("Experiment completed")
print("=" * 60)

print(
    f"Total execution time : "
    f"{total_time:.2f}s"
)

print(
    f"Completed jobs       : "
    f"{completed}"
)

print(
    f"Successful jobs      : "
    f"{successful}"
)

print(
    f"Failed jobs          : "
    f"{failed}"
)

print(
    f"Job results          : "
    f"{JOB_FILE}"
)

print(
    f"Scheduler trace      : "
    f"{TRACE_FILE}"
)

print("=" * 60)