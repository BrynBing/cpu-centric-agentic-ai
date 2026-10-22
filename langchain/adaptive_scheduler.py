"""Independent B/O experiments with fixed admission limits and correlated events.

Run --help for the CLI. The simulation backend requires only the standard library.
The real backend reuses the existing graph stages and requires psutil for cleanup.
"""
import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from scheduling.policy_b import BatchPolicy
from scheduling.policy_o import OverlapPolicy
from scheduling.metrics import percentile
from adaptive_worker import run_request


@dataclass
class Config:
    mode: str = 'dynamic-b'
    backend: str = 'simulate'
    jobs: int = 128
    concurrency: int = 16
    batch_size: int = 8
    batch_grid: tuple = (1, 2, 4, 8, 16, 32, 64)
    prep_slots: int = 8
    tick: float = .02
    sample_interval: float = .5
    timeout: float = 120.0
    latency_target: float = 30.0
    fixed_offset: float = 0.0
    max_hold: float = 5.0
    safety: float = .05
    min_samples: int = 4
    gpu_metrics: bool = False

    def validate(self):
        if self.mode not in ('fixed-b', 'dynamic-b', 'fixed-o', 'dynamic-o', 'serial-o'):
            raise ValueError('unsupported mode')
        if self.backend not in ('simulate', 'real'):
            raise ValueError('unsupported backend')
        if min(self.jobs, self.concurrency, self.batch_size, self.prep_slots,
               self.min_samples) < 1:
            raise ValueError('counts must be positive')
        for name in ('tick', 'sample_interval', 'timeout', 'latency_target', 'max_hold'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('fixed_offset', 'safety'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.min_samples > 32:
            raise ValueError('min_samples cannot exceed predictor history (32)')
        if not self.batch_grid or min(self.batch_grid) < 1:
            raise ValueError('batch grid must contain positive integers')
        if self.mode == 'dynamic-b' and self.batch_size not in self.batch_grid:
            raise ValueError('batch_size must belong to batch_grid')
        if self.mode.endswith('-o'):
            if self.concurrency < 2 * self.batch_size:
                raise ValueError('O requires concurrency >= 2 * batch_size')
            if self.prep_slots < self.batch_size:
                raise ValueError('O requires prep_slots >= batch_size')


def load_requests(path, jobs):
    if path is None:
        rows = [dict(request_id=i + 1,
                     query='Which game won the Spiel des Jahres award most recently?')
                for i in range(jobs)]
    else:
        rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
                if line.strip()]
    if not rows:
        raise ValueError('request manifest is empty')
    ids = set()
    for row in rows:
        identifier = row.get('request_id')
        if not isinstance(identifier, (int, str)) or isinstance(identifier, bool):
            raise ValueError('each request needs a string or integer request_id')
        key = str(identifier)
        if not key or key in ids:
            raise ValueError('request IDs must be nonempty and unique as strings')
        ids.add(key)
        if not isinstance(row.get('query'), str) or not row['query'].strip():
            raise ValueError('each request needs a nonempty query')
        if row.get('arrival', 0) != 0:
            raise ValueError('this harness supports burst arrivals at zero only')
        for field in ('prepare_seconds', 'inference_seconds'):
            value = row.get(field, .05)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{field} must be finite and nonnegative')
        if row.get('fail_stage') not in (None, 'prepare', 'inference'):
            raise ValueError('fail_stage must be prepare or inference')
    return rows


def interval_union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError('negative interval')
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def intersection_duration(left, right):
    left, right = interval_union(left), interval_union(right)
    i = j = 0
    duration = 0.0
    while i < len(left) and j < len(right):
        duration += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return duration


class Runner:
    def __init__(self, config, requests, output):
        config.validate()
        self.cfg = config
        self.requests = requests
        self.output = Path(output)
        try:
            import psutil
            self.psutil = psutil
            psutil.cpu_percent(None)  # Prime a nonblocking CPU sampler.
        except ImportError:
            self.psutil = None
            if config.backend == 'real':
                raise RuntimeError('real runs require psutil (also used by Dynamic N)')
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / 'workers').mkdir()
        self.ctx = mp.get_context('spawn')
        self.gated = config.mode.endswith('-o')
        self.batch = BatchPolicy(config.batch_grid, config.batch_size,
                                 latency_target=config.latency_target) if config.mode == 'dynamic-b' else None
        self.overlap = OverlapPolicy(config.safety, config.max_hold,
                                     min_samples=config.min_samples)
        self.pending = deque(range(len(requests)))
        self.rows = [dict(request_id=r['request_id'], arrival=0.0, status='pending',
                          error='', cohort_id=None, release=None, launch=None, start=None,
                          ready=None, inference_start=None, finish=None, exit=None,
                          latency=None, service_latency=None, queue_wait=None,
                          gate_wait=None, success=False) for r in requests]
        self.workers = {}
        self.cohorts = []
        self.active_b = self.inference = self.successor = None
        self.formed_at = None
        self.events = []
        self.trace = []
        self.cpu = self.gpu = None
        self.cpu_samples = []
        self.abort_reason = None
        self.controller_cpu = 0.0
        self.peak_active = 0
        self.last_decision = None
        self.last_sample = -math.inf

    def event(self, event, now, index=None, cohort=None, **detail):
        row = dict(time=now, event=event,
                   request_id='' if index is None else self.rows[index]['request_id'],
                   cohort_id='' if cohort is None else cohort['id'],
                   detail=json.dumps(detail, sort_keys=True))
        self.events.append(row)

    def new_cohort(self, size, now):
        members = [self.pending.popleft() for _ in range(min(size, len(self.pending)))]
        cohort = dict(id=len(self.cohorts) + 1, target=size, members=members,
                      waiting=deque(members), release=now, ready=None, gate=None,
                      finish=None, clean=True)
        for index in members:
            self.rows[index].update(cohort_id=cohort['id'], release=now)
        self.cohorts.append(cohort)
        self.event('cohort_release', now, cohort=cohort, target=size, actual=len(members))
        return cohort

    def launch(self, cohort, now):
        while cohort['waiting'] and len(self.workers) < self.cfg.concurrency:
            now = time.monotonic() - self.started
            index = cohort['waiting'].popleft()
            parent, child = self.ctx.Pipe()
            process = self.ctx.Process(target=run_request,
                                       args=(child, self.requests[index], self.gated,
                                             self.cfg.backend,
                                             str(self.output / 'workers' / f'{index + 1}.log')))
            self.rows[index].update(launch=now, status='running')
            try:
                process.start()
            except Exception as exc:
                child.close(); parent.close()
                process.close()
                self.finish(index, now, False, f'launch: {exc}')
                self.rows[index]['exit'] = now
                cohort['clean'] = False
                continue
            child.close()
            self.workers[index] = dict(process=process, pipe=parent, eof=False)
            self.peak_active = max(self.peak_active, len(self.workers))
            self.event('launch', now, index, cohort, pid=process.pid)

    def finish(self, index, now, success, error=''):
        row = self.rows[index]
        if row['finish'] is not None:
            return
        row.update(finish=now, latency=now, success=success,
                   status='success' if success else 'failed', error=error)
        if row['start'] is not None:
            row.update(service_latency=now - row['start'], queue_wait=row['start'])
        if row['ready'] is not None and row['inference_start'] is not None:
            row['gate_wait'] = row['inference_start'] - row['ready']
        self.event('terminal', now, index, success=success, error=error)

    def stop_worker(self, worker):
        process = worker['process']
        descendants = []
        if self.psutil and process.pid:
            try:
                descendants = self.psutil.Process(process.pid).children(recursive=True)
            except self.psutil.NoSuchProcess:
                pass
        for child in descendants:
            try:
                child.terminate()
            except self.psutil.NoSuchProcess:
                pass
        if process.is_alive():
            process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill(); process.join(timeout=1)
        if descendants:
            _, alive = self.psutil.wait_procs(descendants, timeout=1)
            for child in alive:
                try:
                    child.kill()
                except self.psutil.NoSuchProcess:
                    pass
            self.psutil.wait_procs(alive, timeout=1)
        worker['pipe'].close()
        if not process.is_alive():
            process.close()

    def poll(self, now):
        for index, worker in list(self.workers.items()):
            pipe, process = worker['pipe'], worker['process']
            def readable():
                if worker['eof']:
                    return False
                try:
                    return pipe.poll()
                except (EOFError, OSError):
                    worker['eof'] = True
                    return False
            while readable():
                try:
                    message = pipe.recv()
                except (EOFError, OSError):
                    worker['eof'] = True
                    break
                stamp = message.pop('time') - self.started
                kind = message.pop('event')
                row = self.rows[index]
                self.event(kind, stamp, index, **message)
                if kind in ('start', 'ready', 'inference_start'):
                    row[kind] = stamp
                if kind == 'done':
                    self.finish(index, stamp, message['success'], message.get('error', ''))
                    if not message['success'] and (row['inference_start'] is not None
                                                  or message.get('phase') == 'inference'):
                        # A failed HTTP call does not prove server work drained.
                        if self.cfg.backend == 'real':
                            self.abort_reason = 'inference failure; server drain unknown'
            if now - self.rows[index]['launch'] >= self.cfg.timeout:
                if process.is_alive():
                    self.finish(index, now, False, 'request timeout')
                    self.rows[index].update(status='timeout', success=False, error='request timeout')
                    self.stop_worker(worker)
                    self.rows[index]['exit'] = time.monotonic() - self.started
                    del self.workers[index]
                    if self.cfg.backend == 'real':
                        self.abort_reason = 'worker timeout; verify server drain before another trial'
                    continue
            if not process.is_alive():
                # An exit can race the earlier pipe poll. Drain its final events
                # on the next tick before classifying a missing outcome.
                if readable():
                    continue
                process.join()
                now = time.monotonic() - self.started
                if self.rows[index]['finish'] is None:
                    self.finish(index, now, False, f'worker exited without outcome: {process.exitcode}')
                    if self.cfg.backend == 'real':
                        self.abort_reason = 'worker lost; server drain unknown'
                elif process.exitcode != 0:
                    self.rows[index].update(success=False, status='failed',
                                            error=f'worker exit code: {process.exitcode}')
                    if self.cfg.backend == 'real':
                        self.abort_reason = 'worker crashed after outcome; verify server drain'
                self.rows[index]['exit'] = now
                pipe.close()
                process.close()
                del self.workers[index]

    def complete(self, cohort):
        return (not cohort['waiting'] and
                all(self.rows[i]['exit'] is not None for i in cohort['members']))

    def finalize_cohort(self, cohort, now):
        cohort['finish'] = now
        cohort['clean'] = all(self.rows[i]['success'] for i in cohort['members'])
        self.event('cohort_terminal', now, cohort=cohort, success=cohort['clean'])

    def step_b(self, now):
        if self.active_b:
            self.launch(self.active_b, now)
            if not self.complete(self.active_b):
                return
            cohort = self.active_b
            self.finalize_cohort(cohort, now)
            if self.batch:
                self.batch.observe(cohort['target'], len(cohort['members']), now - cohort['release'],
                                   [self.rows[i]['finish'] - cohort['release'] for i in cohort['members']
                                    if self.rows[i]['success']], not cohort['clean'])
            self.active_b = None
        if self.pending:
            size, reason = (self.batch.choose(len(self.pending), now) if self.batch
                            else (self.cfg.batch_size, 'fixed'))
            self.event('batch_decision', now, target=size, reason=reason)
            self.active_b = self.new_cohort(size, now)
            self.launch(self.active_b, now)

    def step_o(self, now):
        if self.inference and self.complete(self.inference):
            cohort = self.inference
            self.finalize_cohort(cohort, now)
            if cohort['clean'] and len(cohort['members']) == self.cfg.batch_size:
                self.overlap.observe(inference=max(self.rows[i]['finish'] for i in cohort['members'])
                                     - cohort['gate'])
            self.inference = None
        if self.successor:
            cohort = self.successor
            ready = all(self.rows[i]['ready'] is not None or self.rows[i]['exit'] is not None
                        for i in cohort['members'])
            if ready and cohort['ready'] is None:
                cohort['ready'] = max(self.rows[i]['ready'] if self.rows[i]['ready'] is not None
                                      else self.rows[i]['finish'] for i in cohort['members'])
                if all(self.rows[i]['ready'] is not None for i in cohort['members']) and len(cohort['members']) == self.cfg.batch_size:
                    self.overlap.observe(preparation=cohort['ready'] - cohort['release'])
                self.event('cohort_ready', now, cohort=cohort)
            if ready and self.inference is None:
                cohort['gate'] = now
                for index in cohort['members']:
                    if index in self.workers and self.rows[index]['finish'] is None:
                        try:
                            self.workers[index]['pipe'].send('infer')
                        except (BrokenPipeError, OSError):
                            self.finish(index, now, False, 'inference gate pipe closed')
                self.event('inference_gate_open', now, cohort=cohort)
                self.inference, self.successor = cohort, None
                self.formed_at = now
        if not self.pending or self.successor:
            return
        size = min(self.cfg.batch_size, len(self.pending))
        if len(self.workers) + size > self.cfg.concurrency:
            return
        if self.formed_at is None:
            self.formed_at = now
        elapsed = None if self.inference is None else now - self.inference['gate']
        held = now - self.formed_at
        low_r = high_p = None
        if self.cfg.mode == 'serial-o':
            delay, reason = (self.cfg.tick, 'serial_barrier') if self.inference else (0, 'idle')
        elif self.cfg.mode == 'fixed-o':
            # Gate anchor persists after completion: a fixed offset stays fixed.
            previous_gate = self.cohorts[-1]['gate'] if self.cohorts else None
            delay = max(0, (previous_gate + self.cfg.fixed_offset - now)) if previous_gate is not None else 0
            delay, reason = min(delay, max(0, self.cfg.max_hold - held)), 'fixed_offset'
        else:
            delay, reason, low_r, high_p = self.overlap.delay(elapsed, held)
        decision = (reason, delay == 0)
        if decision != self.last_decision:
            self.event('release_decision', now, delay=delay, reason=reason,
                       remaining_client_drain_lower=low_r, preparation_upper=high_p)
            self.last_decision = decision
        if delay <= 0:
            self.successor = self.new_cohort(self.cfg.batch_size, now)
            self.successor['release_wait'] = held
            self.launch(self.successor, now)
            self.formed_at = None
            self.last_decision = None

    def sample(self, now):
        if now - self.last_sample < self.cfg.sample_interval:
            return
        self.last_sample = now
        self.cpu = self.psutil.cpu_percent(None) / 100 if self.psutil else None
        if self.batch:
            self.batch.sample_cpu(self.cpu)
        if self.cfg.gpu_metrics:
            try:
                result = subprocess.run(['nvidia-smi', '-i', '0',
                                         '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                        capture_output=True, text=True, timeout=.3, check=True)
                self.gpu = float(result.stdout.strip()) / 100
            except (OSError, ValueError, subprocess.SubprocessError):
                self.gpu = None
        self.trace.append(dict(time=now, cpu_util=self.cpu, gpu_util=self.gpu,
                               active=len(self.workers), pending=len(self.pending),
                               n_cap=self.cfg.concurrency,
                               b_target=self.batch.target if self.batch else self.cfg.batch_size))

    def run(self):
        self.started = time.monotonic()
        interrupted = None
        try:
            while self.pending or self.workers or self.active_b or self.inference or self.successor:
                cpu_start = time.thread_time()
                now = time.monotonic() - self.started
                self.poll(now)
                self.sample(now)
                if self.abort_reason:
                    break
                # Polling/telemetry may take time; gates must never precede the
                # ready event they just observed.
                now = time.monotonic() - self.started
                if self.gated:
                    self.step_o(now)
                else:
                    self.step_b(now)
                self.controller_cpu += time.thread_time() - cpu_start
                if self.pending or self.workers:
                    time.sleep(self.cfg.tick)
        except BaseException as exc:
            self.abort_reason = f'{type(exc).__name__}: {exc}'
            interrupted = exc
        finally:
            for index, worker in list(self.workers.items()):
                self.stop_worker(worker)
                now = time.monotonic() - self.started
                self.finish(index, now, False, self.abort_reason or 'cancelled')
                self.rows[index]['exit'] = now
            self.workers.clear()
            now = time.monotonic() - self.started
            for index, row in enumerate(self.rows):
                if row['finish'] is None:
                    self.finish(index, now, False, self.abort_reason or 'cancelled')
                    row['status'] = 'cancelled'
            summary = self.write_results(now)
        if interrupted:
            raise interrupted
        return summary

    def write_results(self, makespan):
        def csv_file(name, rows):
            if not rows:
                return
            with (self.output / name).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        csv_file('requests.csv', self.rows)
        csv_file('events.csv', self.events)
        csv_file('trace.csv', self.trace)
        csv_file('cohorts.csv', [dict(cohort_id=c['id'], target=c['target'],
                                    actual=len(c['members']), release=c['release'],
                                    ready=c['ready'], gate=c['gate'], finish=c['finish'],
                                    release_wait=c.get('release_wait', 0), success=c['clean'])
                               for c in self.cohorts])
        successes = [r for r in self.rows if r['success']]
        preparation = [(r['start'], r['ready']) for r in self.rows
                       if r['start'] is not None and r['ready'] is not None]
        inference = [(r['inference_start'], r['finish']) for r in self.rows
                     if r['inference_start'] is not None]
        overlap = intersection_duration(preparation, inference)
        union = sum(b-a for a,b in interval_union(preparation + inference))
        def utilization(field):
            weighted = duration = 0.0
            for index, row in enumerate(self.trace):
                end = self.trace[index + 1]['time'] if index + 1 < len(self.trace) else makespan
                if row[field] is not None:
                    delta = end - row['time']
                    weighted += row[field] * delta
                    duration += delta
            return weighted / duration if duration else None
        summary = dict(mode=self.cfg.mode, backend=self.cfg.backend, offered=len(self.rows),
                       successful=len(successes), failed=len(self.rows)-len(successes),
                       makespan=makespan, goodput=len(successes)/max(makespan, 1e-9),
                       p50_latency=percentile([r['latency'] for r in successes], .5),
                       p90_latency=percentile([r['latency'] for r in successes], .9),
                       p90_service=percentile([r['service_latency'] for r in successes], .9),
                       preparation_client_overlap_seconds=overlap,
                       preparation_client_overlap_ratio=overlap/union if union else None,
                       gpu_idle_seconds=None, actual_cpu_gpu_overlap_seconds=None,
                       controller_cpu_seconds=self.controller_cpu,
                       mean_cpu_util=utilization('cpu_util'), mean_gpu_util=utilization('gpu_util'),
                       cohort_release_wait_seconds=sum(c.get('release_wait', 0) for c in self.cohorts),
                       peak_active_parents=self.peak_active, aborted=self.abort_reason)
        manifest = dict(config=asdict(self.cfg), python=sys.version,
                        requests_sha256=hashlib.sha256(json.dumps(self.requests, sort_keys=True).encode()).hexdigest(),
                        psutil_available=self.psutil is not None,
                        metric_note='Preparation/client wall overlap is not CPU/GPU compute overlap.',
                        server_observation='client responses only; no server or kernel tracing',
                        server_drain_uncertain=self.abort_reason is not None,
                        request_timeout_includes_startup_and_gate_wait=True)
        for name, data in [('summary.json', summary), ('config.json', manifest),
                           ('manifest.json', self.requests)]:
            (self.output / name).write_text(json.dumps(data, indent=2), encoding='utf-8')
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['fixed-b', 'dynamic-b', 'fixed-o', 'dynamic-o', 'serial-o'], required=True)
    parser.add_argument('--backend', choices=['simulate', 'real'], default='simulate')
    parser.add_argument('--output-dir', type=Path, required=True, help='new directory; never overwrites a trial')
    parser.add_argument('--manifest', type=Path, help='JSONL: request_id, query; optional simulation durations')
    for flag, default in [('jobs',128), ('concurrency',16), ('batch-size',8), ('prep-slots',8), ('min-samples',4)]:
        parser.add_argument('--'+flag, type=int, default=default)
    for flag, default in [('tick',.02), ('sample-interval',.5), ('timeout',120), ('latency-target',30),
                          ('fixed-offset',0), ('max-hold',5), ('safety',.05)]:
        parser.add_argument('--'+flag, type=float, default=default)
    parser.add_argument('--batch-grid', default='1,2,4,8,16,32,64')
    parser.add_argument('--gpu-metrics', action='store_true')
    args = vars(parser.parse_args(argv))
    output, manifest = args.pop('output_dir'), args.pop('manifest')
    try:
        args['batch_grid'] = tuple(int(b) for b in args['batch_grid'].split(','))
        config = Config(**args); config.validate()
        requests = load_requests(manifest, config.jobs)
        summary = Runner(config, requests, output).run()
    except (ValueError, RuntimeError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 1 if summary['failed'] or summary['aborted'] else 0


if __name__ == '__main__':
    sys.exit(main())
