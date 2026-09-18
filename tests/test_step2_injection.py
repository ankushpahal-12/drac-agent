"""
Verification Step 2: Test Chaos Fault Injection Proxy (Runtime Interception).
Verifies that individual faults are correctly armed, intercepted, and recorded.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drac.types import FaultType
from injector.proxy import RuntimeFaultProxy
from agents.calculator_agent import CalculatorAgent
from agents.search_agent import SearchAgent
from agents.db_agent import DatabaseAgent
from agents.multi_agent_pipeline import MultiAgentPipeline

def test_fault_injection():
    print("=======================================================")
    print(" STEP 2: VERIFYING CHAOS FAULT INJECTION PROXY         ")
    print("=======================================================")
    proxy = RuntimeFaultProxy()

    # Fault 1: Tool Timeout (504) on Calculator
    proxy.arm_fault(FaultType.TOOL_TIMEOUT, trigger_step=1)
    calc = CalculatorAgent(proxy)
    res_calc = calc.execute_task("pow(10, 2)", "Compute 10^2")
    tel_calc = res_calc["telemetry"]
    print(f"[Fault 1] TOOL_TIMEOUT on Calculator:")
    print(f"    Success: {res_calc['success']} (Expected False)")
    print(f"    HTTP Status: {tel_calc.http_status}")
    print(f"    Raw Error: {tel_calc.raw_error}")
    print(f"    Simulated Latency: {tel_calc.latency_ms} ms\n")

    # Fault 2: Invalid Args / SQL Column Error on Database
    proxy.arm_fault(FaultType.TOOL_INVALID_ARGS, trigger_step=1)
    db = DatabaseAgent(proxy)
    res_db = db.execute_task("SELECT user_id, order_total FROM orders", "Query orders")
    tel_db = res_db["telemetry"]
    print(f"[Fault 2] TOOL_INVALID_ARGS on SQL Database:")
    print(f"    Success: {res_db['success']} (Expected False)")
    print(f"    Raw Error: {tel_db.raw_error}\n")

    # Fault 3: Empty Return on Web Search
    proxy.arm_fault(FaultType.TOOL_EMPTY_RETURN, trigger_step=1)
    search = SearchAgent(proxy)
    res_search = search.execute_task("drac latency", "Search DRAC latency")
    tel_search = res_search["telemetry"]
    print(f"[Fault 3] TOOL_EMPTY_RETURN on Search Agent:")
    print(f"    Success: {res_search['success']} (Expected False)")
    print(f"    Tool Result: '{tel_search.tool_result}' (Length: {len(tel_search.tool_result)})\n")

    # Fault 4: Inter-Agent Message Loss in Multi-Agent Pipeline
    proxy.arm_fault(FaultType.COMM_MESSAGE_LOSS, trigger_step=1)
    pipeline = MultiAgentPipeline(proxy)
    res_mas = pipeline.run_pipeline("Synthesize market report")
    tel_mas = res_mas["telemetry"]
    print(f"[Fault 4] COMM_MESSAGE_LOSS on Multi-Agent Pipeline:")
    print(f"    Pipeline Success: {res_mas['success']} (Expected False)")
    print(f"    Failed Agent: {res_mas['failed_agent']}")
    print(f"    Raw Error: {tel_mas.raw_error}")
    print("=======================================================\n")

if __name__ == "__main__":
    test_fault_injection()
