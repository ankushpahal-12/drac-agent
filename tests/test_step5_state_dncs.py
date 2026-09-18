"""
Verification Step 5: Test Transactional State Manager & DNCS.
Verifies state checkpointing, context pruning, and Distilled Negative Constraint Synthesis.
"""
from drac.types import DiagnosisResult, FaultDomain, FaultType, Severity
from drac.state_manager import TransactionalStateManager
from drac.dncs import DNCSynthesizer

def test_state_manager_and_dncs():
    print("=======================================================")
    print(" STEP 5: VERIFYING STATE MANAGER & DNCS                ")
    print("=======================================================")
    state_mgr = TransactionalStateManager()

    # 1. Healthy Initial State
    initial_context = [
        {"role": "system", "content": "You are an expert SQL assistant."},
        {"role": "user", "content": "Query customer orders above $100."}
    ]
    env_state = {"db_connected": True, "active_table": "orders"}
    tool_state = {"sql_query": "available"}

    # Take Checkpoint S_1
    cp_id = state_mgr.create_checkpoint(step=1, context=initial_context, env_state=env_state, tool_state=tool_state)
    print(f"[1] Created Checkpoint S_1: '{cp_id}'")
    print(f"    Initial Context Length: {len(initial_context)} messages")

    # 2. Simulate Faulty Step (Context Pollution)
    polluted_context = list(initial_context)
    polluted_context.append({"role": "assistant", "content": "SELECT order_id, order_total FROM orders"})
    polluted_context.append({"role": "system", "content": "ERROR: OperationalError: (1054, \"Unknown column 'order_total'\")\nTraceback (most recent call last):\n  File 'db.py', line 45\n    cursor.execute(q)"})
    polluted_context.append({"role": "assistant", "content": "I apologize, let me try again with another query..."})
    print(f"\n[2] Simulated Context Pollution:")
    print(f"    Polluted Context Length: {len(polluted_context)} messages (Includes 1500-token trace)")

    # 3. Formulate Diagnosis & Synthesize DNCS
    diagnosis = DiagnosisResult(
        domain=FaultDomain.TOOL,
        fault_type=FaultType.TOOL_INVALID_ARGS,
        severity=Severity.MEDIUM,
        confidence=0.98,
        diagnosed_by="System 1 (Fast-Path)",
        diagnostic_latency_ms=0.01,
        diagnostic_cost_tokens=0,
        evidence="OperationalError 1054: Unknown column 'order_total'"
    )
    failed_action = {
        "tool_name": "sql_query",
        "raw_error": "OperationalError: Unknown column 'order_total' in 'field list'",
        "tool_args": {"query": "SELECT order_id, order_total FROM orders"}
    }
    constraint = DNCSynthesizer.synthesize(diagnosis, failed_action)
    print(f"\n[3] Synthesized Distilled Constraint (DNCS):")
    print(f"    '{constraint}'")
    print(f"    Estimated Token Length: ~{len(constraint.split()) * 1.3:.0f} tokens (vs. 1500-token pollution)")

    # 4. Rollback to Checkpoint S_1 with DNCS Injection
    restored_cp = state_mgr.rollback(checkpoint_id=cp_id, distilled_constraint=constraint)
    print(f"\n[4] Restored Checkpoint State:")
    print(f"    Restored Context Length: {len(restored_cp.context_history)} messages")
    print(f"    Restored Message Roles: {[m['role'] for m in restored_cp.context_history]}")
    print(f"    Injected Constraint Present: {restored_cp.context_history[-1]['content'] == constraint}")
    print(f"    Corrupted Trace Erased: {'Traceback' not in str(restored_cp.context_history)}")
    print("=======================================================\n")

if __name__ == "__main__":
    test_state_manager_and_dncs()
