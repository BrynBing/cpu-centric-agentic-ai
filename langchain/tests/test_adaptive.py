import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scheduling.policy_b import BatchPolicy
from scheduling.policy_o import OverlapPolicy
from scheduling.metrics import percentile
from adaptive_scheduler import Config, Runner, load_requests, intersection_duration
from adaptive_worker import run_request


class PoliciesTest(unittest.TestCase):
    def test_pressure_reduces_target_without_changing_capacity(self):
        policy = BatchPolicy([1, 2, 4], 4)
        policy.sample_cpu(.99)
        self.assertEqual(policy.choose(20), (2, 'cpu_pressure'))
        self.assertEqual(policy.choose(20)[0], 1)
        self.assertEqual(policy.choose(20)[0], 1)

    def test_partial_batches_do_not_train_target(self):
        policy = BatchPolicy([1, 2, 4], 4)
        policy.observe(4, 3, 1, [.5, .7, .9], False)
        self.assertEqual(policy.estimates, {})

    def test_probe_and_failure_hold(self):
        policy = BatchPolicy([1, 2, 4], 2, dwell=1, probe_every=1, min_samples=1)
        policy.observe(2, 2, 1, [.8, 1], False)
        self.assertEqual(policy.choose(10), (1, 'probe'))
        policy.observe(1, 1, 1, [], True)
        self.assertEqual(policy.choose(10), (1, 'failure_hold'))

    def test_confirmed_score_change_and_urgency(self):
        from scheduling.policy_b import Estimate
        policy = BatchPolicy([1, 2], 1, dwell=1, confirmations=2)
        policy.estimates = {1: Estimate(1, 1, 3), 2: Estimate(3, 1, 3)}
        policy.reference = 1
        policy.since_change = 3
        self.assertEqual(policy.choose(10, oldest_age=100), (1, 'latency_urgent'))
        self.assertEqual(policy.choose(10), (1, 'confirm'))
        self.assertEqual(policy.choose(10), (2, 'score'))

    def test_upward_probe_needs_cpu_evidence(self):
        policy = BatchPolicy([1, 2], 1, dwell=1, probe_every=1, min_samples=1)
        policy.observe(1, 1, 1, [1], False)
        self.assertEqual(policy.choose(10)[0], 1)
        for _ in range(3):
            policy.sample_cpu(.1)
        self.assertEqual(policy.choose(10), (2, 'probe'))

    def test_conditional_remaining_time_and_fallback(self):
        policy = OverlapPolicy(safety=.2, max_hold=10, min_samples=2)
        self.assertEqual(policy.delay(None, 0)[0], 0)
        self.assertEqual(policy.delay(0, 0)[1], 'cold_or_tail_fallback')
        for _ in range(4):
            policy.observe(preparation=3, inference=8)
        self.assertAlmostEqual(policy.delay(0, 0)[0], 4.8)
        self.assertEqual(policy.delay(5, 0)[0], 0)
        self.assertEqual(policy.delay(0, 10)[0], 0)
        self.assertEqual(policy.delay(9, 0)[1], 'cold_or_tail_fallback')

    def test_statistics(self):
        self.assertEqual(percentile([], .9), None)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], .9), 3.7)
        self.assertEqual(intersection_duration([(0, 4), (2, 6)], [(3, 5), (4, 8)]), 3)

    def test_invalid_capacity_and_manifest(self):
        with self.assertRaises(ValueError):
            Config(mode='dynamic-o', batch_size=8, concurrency=8).validate()
        with self.assertRaises(ValueError):
            Config(tick=float('nan')).validate()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in [
                dict(request_id=1, query='a'), dict(request_id='1', query='b')]))
            with self.assertRaises(ValueError):
                load_requests(path, 2)


class RuntimeTest(unittest.TestCase):
    def run_trial(self, mode, requests, **overrides):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        config = Config(mode=mode, concurrency=4, batch_size=2, prep_slots=2,
                        tick=.005, timeout=10, min_samples=1, **overrides)
        runner = Runner(config, requests, Path(temp.name) / 'trial')
        summary = runner.run()
        self.assertFalse(runner.workers)
        return runner, summary

    @staticmethod
    def requests(count):
        return [dict(request_id=i, query='test', prepare_seconds=.025,
                     inference_seconds=.15) for i in range(count)]

    def test_b_barrier_capacity_tail_and_outcomes(self):
        runner, summary = self.run_trial('fixed-b', self.requests(5))
        self.assertEqual(summary['successful'], 5)
        self.assertEqual([len(c['members']) for c in runner.cohorts], [2, 2, 1])
        self.assertLessEqual(summary['peak_active_parents'], 4)
        for prior, later in zip(runner.cohorts, runner.cohorts[1:]):
            self.assertGreaterEqual(later['release'], prior['finish'])
        for row in runner.rows:
            self.assertAlmostEqual(row['latency'], row['queue_wait'] + row['service_latency'])

    def test_o_gate_serialization_and_preparation_overlap(self):
        runner, summary = self.run_trial('dynamic-o', self.requests(6))
        self.assertEqual(summary['successful'], 6)
        self.assertLessEqual(summary['peak_active_parents'], 4)
        for prior, later in zip(runner.cohorts, runner.cohorts[1:]):
            self.assertGreaterEqual(later['gate'], prior['finish'])
            self.assertGreaterEqual(later['gate'], later['ready'])
        # Cold-start release can overlap startup/preparation with client inference.
        self.assertLess(runner.cohorts[1]['release'], runner.cohorts[0]['finish'])
        self.assertIsNone(summary['actual_cpu_gpu_overlap_seconds'])

    def test_b_refills_when_cohort_exceeds_capacity(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = Runner(Config(mode='fixed-b', concurrency=2, batch_size=5, tick=.005),
                            self.requests(5), Path(temp) / 'trial')
            summary = runner.run()
            self.assertEqual(summary['successful'], 5)
            self.assertEqual(len(runner.cohorts), 1)
            self.assertEqual(summary['peak_active_parents'], 2)

    def test_fixed_offset_is_preserved_after_previous_completion(self):
        runner, summary = self.run_trial('fixed-o', self.requests(4), fixed_offset=.4)
        self.assertEqual(summary['successful'], 4)
        first, second = runner.cohorts
        self.assertGreaterEqual(second['release'], first['gate'] + .4)
        self.assertGreaterEqual(second['release'], first['finish'])

    def test_simulated_inference_failure_is_not_counted_as_goodput(self):
        requests = self.requests(3)
        requests[0]['fail_stage'] = 'inference'
        runner, summary = self.run_trial('dynamic-o', requests)
        self.assertEqual(summary['successful'], 2)
        self.assertEqual(summary['failed'], 1)
        self.assertEqual(summary['goodput'], 2 / summary['makespan'])

    def test_preparation_failures_do_not_deadlock_gate(self):
        requests = self.requests(4)
        requests[0]['fail_stage'] = requests[1]['fail_stage'] = 'prepare'
        runner, summary = self.run_trial('dynamic-o', requests)
        self.assertEqual(summary['successful'], 2)
        self.assertEqual(summary['failed'], 2)
        self.assertTrue(all(r['finish'] is not None for r in runner.rows))

    def test_serial_o_has_no_cross_cohort_overlap(self):
        runner, summary = self.run_trial('serial-o', self.requests(4))
        self.assertEqual(summary['successful'], 4)
        self.assertGreaterEqual(runner.cohorts[1]['release'], runner.cohorts[0]['finish'])

    def test_timeout_is_failed_and_children_are_reaped(self):
        with tempfile.TemporaryDirectory() as temp:
            requests = self.requests(1)
            requests[0]['prepare_seconds'] = 5
            runner = Runner(Config(mode='fixed-b', batch_size=1, timeout=.3, tick=.01),
                            requests, Path(temp) / 'trial')
            summary = runner.run()
            self.assertEqual(summary['successful'], 0)
            self.assertEqual(runner.rows[0]['status'], 'timeout')
            self.assertIsNone(summary['p50_latency'])
            self.assertFalse(runner.workers)


class WorkerAdapterTest(unittest.TestCase):
    def test_real_adapter_preserves_stage_order_and_gates_inference(self):
        calls, messages = [], []
        class Graph:
            def __init__(self, schema):
                self.nodes = {}
            def add_node(self, name, function):
                self.nodes[name] = function
            def add_edge(self, *args):
                pass
            def set_entry_point(self, *args):
                pass
            def set_finish_point(self, *args):
                pass
            def compile(self):
                return self
            def invoke(self, state, config):
                for name in ('web_search', 'fetch_url', 'summarize', 'final_answer'):
                    state.update(self.nodes[name](state))
                return state
        class Connection:
            def send(self, message):
                messages.append(message)
            def recv(self):
                calls.append('gate')
                return 'infer'
            def close(self):
                pass
        def stage(name, result):
            def invoke(state):
                calls.append(name)
                return result
            return invoke
        workload = types.ModuleType('orchestrator')
        workload.GraphState = dict
        workload.web_search = stage('web_search', {'urls': ['test']})
        workload.fetch_url = stage('fetch_url', {'page_texts': ['text']})
        workload.summarize = stage('summarize', {'summaries': ['summary']})
        workload.final_answer = stage('final_answer', {'final_response': 'answer'})
        package = types.ModuleType('langgraph')
        graph = types.ModuleType('langgraph.graph')
        graph.StateGraph = Graph
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {
                'orchestrator': workload, 'langgraph': package, 'langgraph.graph': graph}):
            run_request(Connection(), dict(request_id=1, query='test'), True,
                        'real', str(Path(temp) / 'worker.log'))
        self.assertEqual(calls, ['web_search', 'fetch_url', 'summarize', 'gate', 'final_answer'])
        self.assertTrue(messages[-1]['success'])
        self.assertEqual(messages[-1]['response_chars'], 6)


if __name__ == '__main__':
    unittest.main()
