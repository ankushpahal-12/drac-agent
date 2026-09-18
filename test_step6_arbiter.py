"""
Verification Step 6: Test DRAC Budget-Constrained Recovery Arbiter (B-POMDP).
Verifies that recovery decisions adapt dynamically based on diagnosis and remaining resources.
"""
from drac.types import (
    DiagnosisResult, FaultDomain, FaultType, Severity, RecoveryAction, ExecutionBudget
)
from drac.arbiter import RecoveryArbiter

def test_recovery_arbiter():
    print("=======================================================")
    print(" STEP 6: VERIFYING BUDGET-CONSTRAINED RECOVERY ARBITER ")
    print("=======================================================")
    arbiter = RecoveryArbiter()

    diag_sql = DiagnosisResult(
        domain=FaultDomain.TOOL,
        fault_type=FaultType.TOOL_INVALID_ARGS,
        severity=Severity.MEDIUM,
        confidence=0.97,
        diagnosed_by="System 1 (Fast-Path)",
        diagnostic_latency_ms=0.01,
        diagnostic_cost_tokens=0,
        evidence="OperationalError 1054"
    )

    # Case 1: Abundant Budget (5,000 tokens, 30s)
    budget_abundant = ExecutionBudget(max_tokens=5000, max_time_seconds=30.0, tokens_consumed=200, time_consumed=1.5)
    action_1, util_1 = arbiter.select_action(diag_sql, budget_abundant)
    print(f"[Case 1: Abundant Budget]")
    print(f"    Remaining Budget: {budget_abundant.tokens_remaining} tokens, {budget_abundant.time_remaining:.1f}s")
    print(f"    Selected Action: {action_1.value}")
    print(f"    Calculated Utility: {util_1:.4f} (Expected: ROLLBACK_WITH_DNCS)\n")

    # Case 2: Transient 504 Timeout
    diag_timeout = DiagnosisResult(
        domain=FaultDomain.TOOL,
        fault_type=FaultType.TOOL_TIMEOUT,
        severity=Severity.LOW,
        confidence=0.99,
        diagnosed_by="System 1 (Fast-Path)",
        diagnostic_latency_ms=0.01,
        diagnostic_cost_tokens=0,
        evidence="HTTP 504"
    )
    action_2, util_2 = arbiter.select_action(diag_timeout, budget_abundant)
    print(f"[Case 2: Transient 504 Timeout]")
    print(f"    Selected Action: {action_2.value}\n")

    # Case 3: Exhausted Budget / Critical Safety Stop
    budget_exhausted = ExecutionBudget(max_tokens=2000, max_time_seconds=10.0, tokens_consumed=1950, time_consumed=9.5)
    action_3, util_3 = arbiter.select_action(diag_sql, budget_exhausted)
    print(f"[Case 3: Exhausted Budget (<100 tokens left)]")
    print(f"    Remaining Tokens: {budget_exhausted.tokens_remaining}")
    print(f"    Selected Action: {action_3.value} (Expected: HUMAN_ESCALATION)\n")

    # Case 4: Repeated Failures (3 consecutive failures)
    action_4, util_4 = arbiter.select_action(diag_sql, budget_abundant, consecutive_failures=3)
    print(f"[Case 4: Safety Limit (3 Consecutive Failures)]")
    print(f"    Selected Action: {action_4.value} (Expected: HUMAN_ESCALATION)")
    print("=======================================================")

if __name__ == "__main__":
    test_recovery_arbiter()
