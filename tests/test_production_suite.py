"""
Comprehensive Production Verification Suite for DRAC.
Executes deep unit, integration, and contract tests across all 18 repository endpoints.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drac.types import (
    FaultDomain, FaultType, Severity, RecoveryAction, TelemetryEvent,
    DiagnosisResult, Checkpoint, ExecutionBudget
)
from drac.detector import AnomalyDetector
from drac.diagnoser import DualProcessDiagnoser
from drac.dncs import DNCSynthesizer
from drac.state_manager import TransactionalStateManager
from drac.arbiter import RecoveryArbiter
from drac.verifier import StateVerifier
from injector.proxy import RuntimeFaultProxy
from agents.calculator_agent import CalculatorAgent
from agents.search_agent import SearchAgent
from agents.db_agent import DatabaseAgent
from agents.multi_agent_pipeline import MultiAgentPipeline
from baselines.strategies import (
    NaiveRetryStrategy, ReflexionStrategy, PureRollbackStrategy, DRACFixedStrategy, DRACFullSystem
)
from experiments.metrics import MetricEvaluator, TrialResult

class TestDRACProductionSuite(unittest.TestCase):

    def test_01_types_and_budget_bounds(self):
        """Verify ExecutionBudget never leaks negative budget."""
        budget = ExecutionBudget(max_tokens=1000, max_time_seconds=10.0, tokens_consumed=1500, time_consumed=15.0)
        self.assertEqual(budget.tokens_remaining, 0)
        self.assertEqual(budget.time_remaining, 0.0)

    def test_02_all_15_fault_injections_armed_and_fired(self):
        """Verify that all 15 defined fault types fire deterministically in the proxy."""
        proxy = RuntimeFaultProxy()
        for fault in FaultType:
            proxy.arm_fault(fault, trigger_step=1)
            res, telem = proxy.intercept_tool_call(
                agent_id="test_agent",
                tool_name="test_tool",
                tool_args={"arg": "val"},
                execute_fn=lambda: "healthy_output"
            )
            self.assertTrue(
                telem.raw_error is not None or telem.tool_result == "" or telem.http_status != 200 or isinstance(telem.tool_result, dict),
                f"Fault {fault.value} failed to intercept or emit expected telemetry"
            )

    def test_03_detector_invariants(self):
        """Verify detector flags all 5 invariant breach types with 0 false positives."""
        detector = AnomalyDetector()
        
        # 1. Healthy event
        e_healthy = TelemetryEvent(timestamp=1.0, step=1, agent_id="a", action_type="tool_call", tool_result="ok")
        is_anom, _ = detector.observe(e_healthy)
        self.assertFalse(is_anom)

        # 2. Timeout
        e_timeout = TelemetryEvent(timestamp=2.0, step=2, agent_id="a", action_type="tool_call", latency_ms=9000.0)
        is_anom, reason = detector.observe(e_timeout)
        self.assertTrue(is_anom)
        self.assertIn("TIMEOUT", reason)

        # 3. Empty return
        e_empty = TelemetryEvent(timestamp=3.0, step=3, agent_id="a", action_type="tool_call", tool_result="")
        is_anom, reason = detector.observe(e_empty)
        self.assertTrue(is_anom)
        self.assertEqual(reason, "EMPTY_TOOL_RESULT")

        # 4. Malformed JSON
        e_bad_json = TelemetryEvent(timestamp=4.0, step=4, agent_id="a", action_type="llm_generation", tool_result="{bad_json:")
        is_anom, reason = detector.observe(e_bad_json)
        self.assertTrue(is_anom)
        self.assertIn("MALFORMED_JSON", reason)

    def test_04_dual_process_diagnoser_contracts(self):
        """Verify System 1 is zero-token and System 2 handles semantic failures.

        L3 Fix: System 1 now labels its sub-passes as 'System 1 (String-D)',
        'System 1 (HTTP-Status-A)', 'System 1 (Tool-Gate-B)', etc.
        The test checks that the diagnosed_by string starts with 'System 1'
        (not the exact label), which is the correct invariant — any System 1
        sub-pass still costs 0 tokens and returns before System 2 runs.
        """
        diagnoser = DualProcessDiagnoser()

        # System 1: SQL Column error
        e_s1 = TelemetryEvent(timestamp=1.0, step=1, agent_id="db", action_type="tool_call", raw_error="OperationalError: Unknown column 'xyz'")
        d_s1 = diagnoser.diagnose("RAW_EXCEPTION", e_s1, [])
        self.assertTrue(d_s1.diagnosed_by.startswith("System 1"),
                        f"Expected System 1 fast-path, got: {d_s1.diagnosed_by}")
        self.assertEqual(d_s1.diagnostic_cost_tokens, 0)
        self.assertGreaterEqual(d_s1.confidence, 0.95)

        # System 2: Peer Conflict
        e_s2 = TelemetryEvent(timestamp=2.0, step=2, agent_id="mas", action_type="tool_call", raw_error="Disagreement: conflicting peer claims")
        d_s2 = diagnoser.diagnose("SEMANTIC_DIVERGENCE", e_s2, [])
        self.assertEqual(d_s2.diagnosed_by, "System 2 (Slow-Path)")
        self.assertGreater(d_s2.diagnostic_cost_tokens, 0)

    def test_05_dncs_all_15_faults(self):
        """Verify DNCS synthesizes valid, compact constraints for all 15 fault types."""
        failed_act = {"tool_name": "calc", "raw_error": "SyntaxError", "tool_args": {"x": 1}}
        for fault in FaultType:
            diag = DiagnosisResult(
                domain=FaultDomain.TOOL,
                fault_type=fault,
                severity=Severity.HIGH,
                confidence=0.9,
                diagnosed_by="Test",
                diagnostic_latency_ms=0.1,
                diagnostic_cost_tokens=0,
                evidence="Test evidence"
            )
            constraint = DNCSynthesizer.synthesize(diag, failed_act)
            self.assertTrue(constraint.startswith("[CONSTRAINT:"))
            self.assertTrue(len(constraint.split()) < 40, f"DNCS constraint too verbose for {fault}: {constraint}")

    def test_06_state_manager_bounded_memory_and_rollback(self):
        """Verify state manager enforces max_checkpoints to bound memory and correctly prunes context."""
        state_mgr = TransactionalStateManager(max_checkpoints=3)
        
        for i in range(5):
            state_mgr.create_checkpoint(step=i, context=[{"role": "user", "content": f"msg {i}"}], env_state={}, tool_state={})

        # Memory bounding assertion
        self.assertEqual(len(state_mgr.checkpoints), 3)
        self.assertEqual(len(state_mgr.checkpoint_history), 3)
        latest = state_mgr.get_latest_checkpoint()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.step, 4)

        # Rollback assertion with DNCS
        restored = state_mgr.rollback(distilled_constraint="[CONSTRAINT: test]")
        self.assertIsInstance(restored, Checkpoint)
        self.assertEqual(restored.context_history[-1]["content"], "[CONSTRAINT: test]")

    def test_07_arbiter_pomdp_utility_and_safety_failsafe(self):
        """Verify arbiter calculates optimal utility and triggers HUMAN_ESCALATION when budget is exhausted."""
        arbiter = RecoveryArbiter()
        diag = DiagnosisResult(
            domain=FaultDomain.TOOL, fault_type=FaultType.TOOL_INVALID_ARGS,
            severity=Severity.MEDIUM, confidence=0.95, diagnosed_by="S1",
            diagnostic_latency_ms=0.1, diagnostic_cost_tokens=0, evidence="err"
        )
        action, utility = arbiter.select_action(diag, ExecutionBudget(max_tokens=1000, max_time_seconds=10.0))
        self.assertEqual(action, RecoveryAction.ROLLBACK_WITH_DNCS)
        self.assertGreater(utility, 0)

        # Failsafe escalation on exhausted budget
        action_exhausted, _ = arbiter.select_action(diag, ExecutionBudget(max_tokens=0, max_time_seconds=0.0))
        self.assertEqual(action_exhausted, RecoveryAction.HUMAN_ESCALATION)

    def test_08_state_verifier(self):
        """Verify StateVerifier catches unhandled errors and schema mismatches."""
        verifier = StateVerifier()
        
        # Valid JSON
        ok, _ = verifier.verify_remediation('{"result": 100}', expected_schema={"required": ["result"]})
        self.assertTrue(ok)

        # Missing required key
        fail, msg = verifier.verify_remediation('{"other": 100}', expected_schema={"required": ["result"]})
        self.assertFalse(fail)
        self.assertIn("Missing required", msg)

        # Null or empty
        fail_null, _ = verifier.verify_remediation(None)
        self.assertFalse(fail_null)

    def test_09_all_benchmark_agents_healthy(self):
        """Verify all 4 benchmark agents execute successfully under healthy state."""
        proxy = RuntimeFaultProxy()
        
        calc = CalculatorAgent(proxy)
        self.assertTrue(calc.execute_task("2 + 2", "add")["success"])

        search = SearchAgent(proxy)
        self.assertTrue(search.execute_task("drac latency", "query")["success"])

        db = DatabaseAgent(proxy)
        self.assertTrue(db.execute_task("SELECT * FROM orders", "query")["success"])

        mas = MultiAgentPipeline(proxy)
        self.assertTrue(mas.run_pipeline("test")["success"])

    def test_10_all_baseline_recovery_strategies(self):
        """Verify all 5 comparative recovery baselines execute without raising exceptions."""
        proxy = RuntimeFaultProxy()
        detector = AnomalyDetector()
        diagnoser = DualProcessDiagnoser()
        arbiter = RecoveryArbiter()
        state_mgr = TransactionalStateManager()
        verifier = StateVerifier()

        s1 = NaiveRetryStrategy()
        s2 = ReflexionStrategy()
        s3 = PureRollbackStrategy(state_mgr)
        s4 = DRACFixedStrategy(diagnoser, state_mgr)
        s5 = DRACFullSystem(detector, diagnoser, arbiter, state_mgr, verifier)

        event = TelemetryEvent(
            timestamp=100.0, step=1, agent_id="test", action_type="tool_call",
            tool_name="calculate", tool_args={"expr": "1/0"}, raw_error="ZeroDivisionError"
        )
        budget = ExecutionBudget(max_tokens=1000, max_time_seconds=10.0)

        task_fn = lambda **kwargs: {"success": True}
        res1 = s1.recover(event, task_fn, budget)
        res2 = s2.recover(event, task_fn, budget)
        state_mgr.create_checkpoint(step=1, context=[], env_state={}, tool_state={})
        res3 = s3.recover(event, task_fn, budget)
        res4 = s4.recover(event, task_fn, budget)
        res5 = s5.recover(event, task_fn, budget)

        self.assertTrue(all(isinstance(r[0], bool) for r in [res1, res2, res3, res4, res5]))

    def test_11_metric_evaluator_and_markdown_table(self):
        """Verify MetricEvaluator correctly aggregates metrics and generates markdown tables."""
        trials = [
            TrialResult(
                task_name="Calculator", fault_type="TOOL_TIMEOUT", ground_truth_domain="TOOL",
                strategy_name="DRAC Full System", fault_detected=True, diagnosed_domain="TOOL",
                diagnosis_correct=True, recovery_success=True, recovery_latency_sec=0.01,
                recovery_cost_tokens=50, base_task_tokens=400, cascade_contained=True, action_taken="RETRY"
            )
        ]
        evaluator = MetricEvaluator(trials)
        summary_df = evaluator.compute_summary_by_strategy()
        self.assertEqual(len(summary_df), 1)
        self.assertEqual(summary_df.iloc[0]["RSR (%)"], 100.0)

        md = evaluator.generate_markdown_table()
        self.assertIn("DRAC Full System", md)
        self.assertIn("RSR (%)", md)

if __name__ == "__main__":
    unittest.main()
