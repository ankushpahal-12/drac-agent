"""
Verification Step 7: Head-to-Head Comparison on a Concrete Fault Instance.
Simulates an SQL Column Error across the 4 recovery paradigms side-by-side.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drac.types import ExecutionBudget, TelemetryEvent
from drac.detector import AnomalyDetector
from drac.diagnoser import DualProcessDiagnoser
from drac.arbiter import RecoveryArbiter
from drac.state_manager import TransactionalStateManager
from drac.verifier import StateVerifier
from baselines.strategies import (
    NaiveRetryStrategy, ReflexionStrategy, PureRollbackStrategy, DRACFullSystem
)

def test_side_by_side_comparison():
    print("=======================================================")
    print(" STEP 7: SIDE-BY-SIDE RECOVERY PARADIGM COMPARISON     ")
    print("=======================================================")
    print("Target Fault: OperationalError: Unknown column 'order_total'\n")

    failed_event = TelemetryEvent(
        timestamp=1.0, step=1, agent_id="db_agent", action_type="tool_call",
        tool_name="sql_query", tool_args={"query": "SELECT user_id, order_total FROM orders"},
        tool_result=None,
        raw_error="OperationalError: (1054, \"Unknown column 'order_total' in 'field list'\")",
        http_status=400, latency_ms=18.0
    )

    # 1. Strategy 1: Naive Retry
    strat1 = NaiveRetryStrategy()
    b1 = ExecutionBudget(max_tokens=4000, max_time_seconds=30.0)
    succ1, cost1, lat1 = strat1.recover(failed_event, None, b1)
    print(f"[Strategy 1: Naive Retry]")
    print(f"    Success: {succ1}")
    print(f"    Tokens Spent: {cost1} | Latency: {lat1:.4f}s")
    print(f"    Context State: Polluted with raw traceback (Negative Priming Risk)\n")

    # 2. Strategy 2: Reflexion (In-band)
    strat2 = ReflexionStrategy()
    b2 = ExecutionBudget(max_tokens=4000, max_time_seconds=30.0)
    succ2, cost2, lat2 = strat2.recover(failed_event, None, b2)
    print(f"[Strategy 2: Reflexion (In-band)]")
    print(f"    Success: {succ2}")
    print(f"    Tokens Spent: {cost2} | Latency: {lat2:.4f}s")
    print(f"    Context State: Polluted with 280-token reflection reasoning\n")

    # 3. Strategy 3: Pure Rollback (Amnesia)
    state_mgr = TransactionalStateManager()
    strat3 = PureRollbackStrategy(state_mgr)
    b3 = ExecutionBudget(max_tokens=4000, max_time_seconds=30.0)
    succ3, cost3, lat3 = strat3.recover(failed_event, None, b3)
    print(f"[Strategy 3: Pure Rollback]")
    print(f"    Success: {succ3}")
    print(f"    Tokens Spent: {cost3} | Latency: {lat3:.4f}s")
    print(f"    Context State: Clean, but Amnesic (No knowledge of what failed)\n")

    # 4. Strategy 4: DRAC Full System
    detector = AnomalyDetector()
    diagnoser = DualProcessDiagnoser()
    arbiter = RecoveryArbiter()
    verifier = StateVerifier()
    strat4 = DRACFullSystem(detector, diagnoser, arbiter, state_mgr, verifier)
    b4 = ExecutionBudget(max_tokens=4000, max_time_seconds=30.0)
    succ4, cost4, lat4, diag4, act4 = strat4.recover(failed_event, None, b4)
    print(f"[Strategy 4: DRAC Full System]")
    print(f"    Diagnosis: {diag4.domain.value} -> {diag4.fault_type.value} via {diag4.diagnosed_by}")
    print(f"    Action Chosen: {act4.value}")
    print(f"    Success: {succ4}")
    print(f"    Tokens Spent: {cost4} | Latency: {lat4:.4f}s")
    print(f"    Context State: Clean Checkpoint + Compact 20-tok Constraint (DNCS)")
    print("=======================================================\n")

if __name__ == "__main__":
    test_side_by_side_comparison()
