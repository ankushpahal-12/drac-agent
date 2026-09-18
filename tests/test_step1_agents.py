"""
Verification Step 1: Test Benchmark Agents under Normal (Healthy) Execution.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from injector.proxy import RuntimeFaultProxy
from agents.calculator_agent import CalculatorAgent
from agents.search_agent import SearchAgent
from agents.db_agent import DatabaseAgent
from agents.multi_agent_pipeline import MultiAgentPipeline

def test_healthy_agents():
    print("=======================================================")
    print(" STEP 1: VERIFYING BENCHMARK AGENTS (HEALTHY STATE)    ")
    print("=======================================================")
    proxy = RuntimeFaultProxy()  # Disarmed: no faults injected

    # 1. Calculator Agent
    calc = CalculatorAgent(proxy)
    expr = "round(pow(14.5, 2) + math.sqrt(225), 2)"
    res_calc = calc.execute_task(expr, "Compute (14.5^2 + sqrt(225))")
    print(f"[1] Calculator Agent:")
    print(f"    Task: {expr}")
    print(f"    Success: {res_calc['success']}")
    print(f"    Result: {res_calc['result']} (Expected: 225.25)\n")

    # 2. Search Agent
    search = SearchAgent(proxy)
    res_search = search.execute_task("agentchaos ase 2026", "Find literature on AgentChaos")
    print(f"[2] Web Search Agent:")
    print(f"    Query: 'agentchaos ase 2026'")
    print(f"    Success: {res_search['success']}")
    print(f"    Result: {res_search['result']}\n")

    # 3. Database Agent
    db = DatabaseAgent(proxy)
    res_db = db.execute_task("SELECT user_id, total_amount FROM orders", "Query order totals")
    print(f"[3] SQL Database Agent:")
    print(f"    SQL Query: 'SELECT user_id, total_amount FROM orders'")
    print(f"    Success: {res_db['success']}")
    print(f"    Rows Returned: {res_db['result']['rows']}\n")

    # 4. Multi-Agent Pipeline
    pipeline = MultiAgentPipeline(proxy)
    res_mas = pipeline.run_pipeline("Synthesize market report")
    print(f"[4] Multi-Agent Pipeline:")
    print(f"    Success: {res_mas['success']}")
    print(f"    Final Report: {res_mas['final_report']}")
    print("=======================================================\n")

if __name__ == "__main__":
    test_healthy_agents()
