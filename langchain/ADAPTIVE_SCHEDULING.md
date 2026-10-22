# Independent Dynamic B and Dynamic O

The new harness implements two independently selectable optimizations. Use
Python 3.9 or newer. The existing N launcher delegates its unchanged threshold
rule to `scheduling/policy_n.py`; its CLI, sampling, admission and CSV behavior
remain the same. The static scheduler and orchestrator are unchanged.

## File organization

All three independent policy modules use the same `policy_<dimension>.py` format:

```text
langchain/
  scheduling/
    __init__.py
    policy_n.py       # N: process admission target, CPU percentage input
    policy_b.py       # B: next cohort size, CPU fraction input
    policy_o.py       # O: next preparation release time
    metrics.py        # Shared statistics, not a control policy
  dynamic_scheduler.py          # Existing N experiment entry point
  adaptive_scheduler.py         # Shared B/O execution and instrumentation
  adaptive_worker.py            # Request execution and inference gate
  run_adaptive_experiments.py    # Paired B/O experiments
  analyze_adaptive_results.py    # Trial-level analysis
```

`adaptive_policies.py` now contains compatibility imports only; implementations
live in the separate modules. New code should import those modules directly.
This refactor separates responsibilities without enabling combined controllers.
Future N+B or combinations with O must coordinate capacity, membership and
release timing in the runtime, including invalidating timing estimates when
capacity or cohort size changes. N uses CPU percentages, while B uses fractions;
any shared state adapter must convert explicitly.

## Run a complete simulation experiment

From the repository root:

```bash
python langchain/run_adaptive_experiments.py --backend simulate --jobs 32 --repeats 3 --output-dir results/adaptive_sim
python langchain/analyze_adaptive_results.py results/adaptive_sim
python -m unittest discover -s langchain/tests -v
```

Each experiment needs a new output directory to prevent accidental overwrites.
Simulation uses sleeping workers and requires only Python's standard library.
It verifies scheduling and instrumentation, **not CPU/GPU performance**. If
psutil is absent, CPU measurements are null and upward B probes are disabled;
they are not driven by fabricated utilization. Install/use psutil in the real
benchmark environment, as required by the existing Dynamic N implementation.

## Run the real workload

Activate the repository's LangChain environment and install psutil if it is not
already present. Start and warm an otherwise dedicated vLLM server using the
repository's model and endpoint: `openai/gpt-oss-20b` at `localhost:5000/v1`.
Wait for all warm-up requests to finish before measuring. Keep server settings,
CPU affinity, caches and workload inputs identical across trials.

```bash
python langchain/adaptive_scheduler.py --mode dynamic-b --backend real --jobs 128 --concurrency 16 --batch-size 8 --output-dir results/dynamic_b_01
python langchain/adaptive_scheduler.py --mode dynamic-o --backend real --jobs 128 --concurrency 16 --prep-slots 8 --batch-size 8 --output-dir results/dynamic_o_01
python langchain/run_adaptive_experiments.py --backend real --jobs 256 --repeats 10 --output-dir results/adaptive_real
```

The real worker builds the same four-node LangGraph pipeline using the existing
stage functions. Each admitted parent runs one request, with graph concurrency
one. LexRank still creates its existing child pool; the parent cap is not a cap
on all OS processes or CPU cores. Real runs require psutil for descendant cleanup.
Worker imports and process startup are included in measured request latency.
The default query and skip-web-search behavior match the existing N workload.

`--timeout` bounds parent lifetime including startup, preparation, gate waiting
and inference. An ambiguous real inference error, unexpected worker loss or
timeout aborts the trial and stops the multi-trial experiment. Killing a client
does not cancel server work: inspect/drain or restart the server before another
run. Normal responses are the available completion signal; there is no claim of
kernel-level or server-side cancellation instrumentation.

Use `--gpu-metrics` on a single-trial command to sample GPU 0 utilization with
`nvidia-smi`. Missing/failed samples remain null. Sampling is optional because
launching nvidia-smi can perturb timing; enable it equally for matched runs.

## Request manifests

`--manifest workload.jsonl` accepts one JSON object per request:

```json
{"request_id": "q1", "query": "Which game won the Spiel des Jahres award most recently?", "skip_web_search": true}
{"request_id": "q2", "query": "A second query", "skip_web_search": false}
```

IDs must be unique. Manifest length overrides `--jobs`; its exact normalized
contents and hash are saved per trial. This first implementation supports a
finite burst with all arrivals at time zero; nonzero arrival offsets are rejected.
For simulation, optional `prepare_seconds`, `inference_seconds`, and
`fail_stage` (`prepare` or `inference`) enable timing/failure scenarios. These
simulation controls are not used to execute real stages. With search skipped,
the existing code uses its fixed URLs even for a custom query. Live web content
is not deterministic: use identical captured inputs in a future replay adapter
before making strong performance claims.

## Control policies and matched baselines

| Mode | Adapted variable | Fixed behavior |
|---|---|---|
| fixed-b | None | Constant cohort size, fixed parent cap, completion barrier |
| dynamic-b | Next cohort size | Same cap, FIFO refill within cohort, same barrier |
| fixed-o | None | Fixed cohort and capacities; fixed offset from previous inference gate |
| dynamic-o | Next preparation release time | Same cohort, capacities and inference gate |
| serial-o | None | Next preparation waits for preceding cohort to finish |

B groups queued requests and refills at most `--concurrency` parent slots within
the active group. All parents must exit before the next group begins. It does
not equate group size with server tensor batch size. A small B may underfill the
fixed cap. The final partial cohort is dispatched immediately and excluded from
target-size model learning.

Dynamic B maintains per-size EWMA goodput and P90 release-relative latency on
the grid `--batch-grid`. Its dimensionless score is
`goodput / initial_goodput - 0.25 * p90 / latency_target - 0.02 * abs(log2(candidate/current))`.
It compares adjacent sizes, requires a score improvement over 0.05 for two
decisions and at least three completed cohorts between ordinary changes. It
learns each new target from at least two complete cohorts, probes stale/unknown
neighbors after five cohorts, and expires estimates after twenty cohorts.
Upward probes require three low-CPU samples (smoothed utilization below 65%).
Three high samples (above 85%), or an emergency raw sample above 95%, reduce
the next target one grid step. CPU smoothing uses alpha 0.2; performance EWMA
uses alpha 0.3. A failed cohort blocks speculative switching until a healthy
observation. Oldest-request urgency suppresses exploration at the declared
latency target; this is not an SLA guarantee. Only future groups are changed.

O admits at most one preparation/ready cohort plus one inference cohort. It
requires `concurrency >= 2 * batch_size` and `prep_slots >= batch_size`. Since
only one cohort prepares, this enforces the fixed preparation bound without
changing it dynamically. Every parent waits at the inference gate after its
own preparation; the gate opens only after all members are ready/failed and
the previous inference cohort's parents have exited. Failed preparation
members do not block the remaining members, including an all-failed cohort.

Dynamic O keeps the latest 32 full-cohort observations. Given elapsed inference
time a, it computes historical residuals `g-a` only for `g>a` (conditional
survival). Remaining-time lower envelope is mean minus mean absolute deviation;
preparation upper envelope is mean plus mean absolute deviation. Additional
hold is `max(0, lower_remaining - upper_preparation - safety)`, capped by the
remaining `--max-hold` budget. Startup, an idle inference gate or inadequate
tail history uses immediate feasible release. `--min-samples` defaults to four.
The timer re-evaluates the decision every `--tick` seconds; its default is 0.02 s.

The O estimate is **remaining client cohort completion time**, not GPU execution
time. Its gate changes how requests reach vLLM, so compare it first with fixed-o
on that same architecture. `--fixed-offset` is measured from the preceding
gate opening, including when that cohort already finished. Both policies share
`--max-hold`; serial-o deliberately waits for completion. Report architectural
cost separately from adaptive-policy benefit. B and O are not combined here.

## Recorded metrics

Each trial produces:

- `requests.csv`: IDs, outcomes/errors, monotonic-relative arrival, release,
  launch, worker start, preparation ready, inference start, finish and observed
  exit; total/service latency, queue and gate wait.
- `events.csv`: correlated request/cohort transitions, individual stage start/end
  events, worker CPU time, process ID and decision reason. Worker CPU time does
  not include LexRank descendants. Events are buffered until trial finalization.
- `cohorts.csv`: requested/actual size, release, readiness, gate, terminal time,
  and cohort release waiting.
- `trace.csv`: fixed N, current B, parent occupancy and optional host CPU/GPU
  fractions. Pending counts refer to unassigned requests, not cohort members.
- `summary.json`, `config.json`, `manifest.json`, plus per-worker logs.

Total latency is `finish - arrival`; service latency is `finish - worker_start`;
queue waiting is `worker_start - arrival`. Successful-request goodput is
`successful / makespan`, including unsuccessful work and cleanup in elapsed time.
Success means the pipeline returned a nonempty response, not verified factual
answer correctness. P50/P90 use linear interpolation over successful requests;
zero-success percentiles are null, and failures/cancellations are reported
separately. Never compare success-conditioned latency without failure rate.

Preparation/client overlap is an intersection of **unions** of wall intervals,
avoiding double-counting concurrent requests. The ratio divides by their union.
This includes network waiting and client/server overhead. Actual CPU/GPU compute
overlap and GPU idle time are deliberately null until suitable tracing exists.
Do not interpret these proxies as accelerator measurements. Controller CPU time
excludes sleeps; average utilization is weighted by sample duration.

The experiment runner saves randomized, seeded policy order and `comparison.csv`.
The analysis script saves per-mode means/standard deviations and bootstrapped
paired differences with 95% intervals, resampling trial pairs rather than
individual requests. One pair produces no confidence interval. Three repeats
are a smoke-test default, not sufficient statistical evidence. Use longer
traces and repeated trials; 128 requests may provide too few large cohorts for
adaptation. Online learning/probes are included in measured runtime.

Existing N trials remain available through their original scripts. Their
polling cadence, output outcomes and architecture differ from this harness;
do not combine their CSV rows into a purported matched B/O comparison.
