import argparse
import csv
import os
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--jobs", type=int, default=128)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

N = args.n
TOTAL_JOBS = args.jobs

pending = list(range(1, TOTAL_JOBS + 1))
running = {}
results = {}

start = time.time()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

while pending or running:

    # Check completed jobs
    finished = []

    for job_id, process in running.items():
        if process.poll() is not None:
            finished.append(job_id)

    for job_id in finished:
        finish_time = time.time() - start

        results[job_id]["finish_time"] = finish_time
        results[job_id]["latency"] = finish_time

        del running[job_id]

    # Launch jobs up to fixed N
    while pending and len(running) < N:
        job_id = pending.pop(0)

        launch_time = time.time() - start

        process = subprocess.Popen(
            [
                "python",
                "orchestrator.py",
                "--skip-web-search",
                "--job-id",
                str(job_id),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        running[job_id] = process

        results[job_id] = {
            "launch_time": launch_time
        }

    print(
        f"N={N} | "
        f"Running={len(running)} | "
        f"Pending={len(pending)} | "
        f"Completed={len(results) - len(running)}"
    )

    time.sleep(0.2)


with open(args.output, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "job_id",
        "launch_time",
        "finish_time",
        "latency"
    ])

    for job_id in sorted(results):
        r = results[job_id]

        writer.writerow([
            job_id,
            r["launch_time"],
            r["finish_time"],
            r["latency"]
        ])

print(f"Results saved to {args.output}")