"""
Verification Step 4: Test DRAC Dual-Process Root-Cause Diagnoser.
Verifies System 1 Fast-Path (0 cost, 0 ms) and System 2 Slow-Path classification.
"""
from drac.types import TelemetryEvent, FaultDomain, FaultType
from drac.diagnoser import DualProcessDiagnoser

def test_dual_process_diagnoser():
    print("=======================================================")
    print(" STEP 4: VERIFYING DUAL-PROCESS DIAGNOSER (S1 & S2)    ")
    print("=======================================================")
    diagnoser = DualProcessDiagnoser()

    # Test Case 1: SQL Column Error (System 1 Fast-Path)
    e1 = TelemetryEvent(
        timestamp=1.0, step=1, agent_id="db", action_type="tool_call",
        tool_name="sql_query", raw_error="OperationalError: (1054, \"Unknown column 'order_total'\")"
    )
    d1 = diagnoser.diagnose("RAW_EXCEPTION", e1, [])
    print(f"[Case 1: SQL Unknown Column]")
    print(f"    Diagnosed Domain: {d1.domain.value}")
    print(f"    Diagnosed Fault: {d1.fault_type.value}")
    print(f"    Engine: {d1.diagnosed_by}")
    print(f"    Confidence: {d1.confidence}")
    print(f"    Token Cost: {d1.diagnostic_cost_tokens} tokens (Expected 0)")
    print(f"    Latency: {d1.diagnostic_latency_ms:.3f} ms\n")

    # Test Case 2: Gateway Timeout 504 (System 1 Fast-Path)
    e2 = TelemetryEvent(
        timestamp=2.0, step=1, agent_id="calc", action_type="tool_call",
        tool_name="calc_eval", http_status=504, latency_ms=8500.0
    )
    d2 = diagnoser.diagnose("EXECUTION_TIMEOUT_8500ms", e2, [])
    print(f"[Case 2: 504 Gateway Timeout]")
    print(f"    Diagnosed Domain: {d2.domain.value}")
    print(f"    Diagnosed Fault: {d2.fault_type.value}")
    print(f"    Engine: {d2.diagnosed_by}")
    print(f"    Token Cost: {d2.diagnostic_cost_tokens} tokens (Expected 0)\n")

    # Test Case 3: Malformed JSON (System 1 Fast-Path)
    e3 = TelemetryEvent(
        timestamp=3.0, step=1, agent_id="llm", action_type="llm_generation",
        tool_result='{"unclosed": "brace', raw_error="JSONDecodeError: Unterminated string"
    )
    d3 = diagnoser.diagnose("MALFORMED_JSON_SCHEMA", e3, [])
    print(f"[Case 3: Malformed JSON Schema]")
    print(f"    Diagnosed Domain: {d3.domain.value}")
    print(f"    Diagnosed Fault: {d3.fault_type.value}")
    print(f"    Engine: {d3.diagnosed_by}\n")

    # Test Case 4: Stale Context State (System 2 Slow-Path)
    e4 = TelemetryEvent(
        timestamp=4.0, step=2, agent_id="planner", action_type="tool_call",
        tool_name="state_fetch", raw_error="ContextWarning: stale cached state returned from memory"
    )
    d4 = diagnoser.diagnose("SEMANTIC_DIVERGENCE", e4, [])
    print(f"[Case 4: Stale Context Memory]")
    print(f"    Diagnosed Domain: {d4.domain.value}")
    print(f"    Diagnosed Fault: {d4.fault_type.value}")
    print(f"    Engine: {d4.diagnosed_by}")
    print(f"    Token Cost: {d4.diagnostic_cost_tokens} tokens (Expected >0)")
    print(f"    Evidence: {d4.evidence}\n")

    # Test Case 5: Conflicting Peer in Multi-Agent (System 2 Slow-Path)
    e5 = TelemetryEvent(
        timestamp=5.0, step=3, agent_id="analyst", action_type="tool_call",
        tool_name="inter_agent", raw_error="Disagreement: conflicting peer claims regarding metric A"
    )
    d5 = diagnoser.diagnose("SEMANTIC_DIVERGENCE", e5, [])
    print(f"[Case 5: Multi-Agent Peer Conflict]")
    print(f"    Diagnosed Domain: {d5.domain.value}")
    print(f"    Diagnosed Fault: {d5.fault_type.value}")
    print(f"    Engine: {d5.diagnosed_by}")
    print(f"    Token Cost: {d5.diagnostic_cost_tokens} tokens (Expected >0)")
    print("=======================================================\n")

if __name__ == "__main__":
    test_dual_process_diagnoser()
