
"""Run paired adaptive scheduling experiments, analyze, and plot results."""

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

from adaptive_scheduler import Config, Runner, load_requests


def run_analysis_and_plots(output_dir, family):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    analyzer = script_dir / "analyze_adaptive_results.py"
    plotter = project_root / "plot_b_mechanisms.py"

    print("\n=== Generating aggregate analysis ===", flush=True)

    subprocess.run(
        [sys.executable, str(analyzer), str(output_dir)],
        check=True,
        cwd=project_root,
    )

    if family in ("b", "both"):
        if not plotter.is_file():
            raise FileNotFoundError(
                f"Plotting script not found: {plotter}"
            )

        print("\n=== Generating B scheduling figures ===", flush=True)

        subprocess.run(
            [
                sys.executable,
                str(plotter),
                "--results-dir",
                str(output_dir),
            ],
            check=True,
            cwd=project_root,
        )

        print(
            "\nFigures saved to:",
            output_dir / "figures_mechanisms",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--backend",
        choices=["simulate", "real"],
        default="simulate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--jobs", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fixed-offset", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--batch-grid",
        default="1,2,4,8,16,32,64",
    )
    parser.add_argument("--latency-target", type=float, default=30)
    parser.add_argument("--max-hold", type=float, default=5)
    parser.add_argument("--safety", type=float, default=0.05)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--gpu-metrics", action="store_true")
    parser.add_argument(
        "--family",
        choices=["b", "o", "both"],
        default="both",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Run experiments and aggregation without figures",
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("repeats must be positive")

    args.output_dir = args.output_dir.resolve()

    requests = load_requests(args.manifest, args.jobs)

    modes = []

    if args.family in ("b", "both"):
        modes += ["fixed-b", "dynamic-b"]

    if args.family in ("o", "both"):
        modes += ["fixed-o", "dynamic-o", "serial-o"]

    configs = {
        mode: Config(
            mode=mode,
            backend=args.backend,
            jobs=len(requests),
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            prep_slots=args.batch_size,
            fixed_offset=args.fixed_offset,
            batch_grid=tuple(
                int(b) for b in args.batch_grid.split(",")
            ),
            latency_target=args.latency_target,
            max_hold=args.max_hold,
            safety=args.safety,
            min_samples=args.min_samples,
            gpu_metrics=args.gpu_metrics,
            timeout=args.timeout,
        )
        for mode in modes
    }

    for config in configs.values():
        config.validate()

    args.output_dir.mkdir(parents=True, exist_ok=False)

    rng = random.Random(args.seed)
    schedule = []

    for repeat in range(args.repeats):
        order = modes[:]
        rng.shuffle(order)
        schedule.extend(
            (repeat + 1, mode)
            for mode in order
        )

    (args.output_dir / "schedule.json").write_text(
        json.dumps(schedule, indent=2),
        encoding="utf-8",
    )

    results = []

    for repeat, mode in schedule:
        print(f"\n=== Trial {repeat}: {mode} ===", flush=True)

        summary = Runner(
            configs[mode],
            requests,
            args.output_dir / f"{repeat:03d}_{mode}",
        ).run()

        results.append(dict(repeat=repeat, **summary))

        with (args.output_dir / "comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(results[0]),
            )
            writer.writeheader()
            writer.writerows(results)

        if summary["aborted"]:
            raise RuntimeError(
                "Trial aborted. Remaining trials stopped; "
                "inspect logs and drain server."
            )

    print("\n=== Experiments completed ===", flush=True)

    project_root = Path(__file__).resolve().parent.parent
    analyzer = (
        Path(__file__).resolve().parent
        / "analyze_adaptive_results.py"
    )

    subprocess.run(
        [
            sys.executable,
            str(analyzer),
            str(args.output_dir),
        ],
        check=True,
        cwd=project_root,
    )

    if not args.skip_plots and args.family in ("b", "both"):
        plotter = project_root / "plot_b_mechanisms.py"

        if not plotter.is_file():
            raise FileNotFoundError(
                f"Plotting script not found: {plotter}"
            )

        print("\n=== Generating figures ===", flush=True)

        subprocess.run(
            [
                sys.executable,
                str(plotter),
                "--results-dir",
                str(args.output_dir),
            ],
            check=True,
            cwd=project_root,
        )

    print("\n=== Workflow completed ===")
    print("Results:", args.output_dir)

    return 1 if any(row["failed"] for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
