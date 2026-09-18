"""
Benchmark Agent 1: Scientific Calculator & Math Tool Agent.
Performs multi-step arithmetic, unit conversions, and algebraic evaluations.
"""
from typing import Dict, Any, List
import math
from injector.proxy import RuntimeFaultProxy

class CalculatorAgent:
    def __init__(self, proxy: RuntimeFaultProxy):
        self.proxy = proxy
        self.memory: Dict[str, Any] = {}
        self.context: List[Dict[str, Any]] = []

    def execute_task(self, expression: str, task_prompt: str) -> Dict[str, Any]:
        self.context.append({"role": "user", "content": task_prompt})
        
        def _calc():
            # Safe evaluation
            allowed = {"math": math, "abs": abs, "round": round, "pow": pow, "sqrt": math.sqrt}
            return eval(expression, {"__builtins__": None}, allowed)

        res, telemetry = self.proxy.intercept_tool_call(
            agent_id="calculator_agent",
            tool_name="calculator_eval",
            tool_args={"expression": expression},
            execute_fn=_calc
        )

        if telemetry.raw_error or res is None:
            self.context.append({"role": "assistant", "content": f"Tool error: {telemetry.raw_error}"})
            return {"success": False, "result": None, "telemetry": telemetry}

        self.memory["last_result"] = res
        self.context.append({"role": "assistant", "content": f"Result: {res}"})
        return {"success": True, "result": res, "telemetry": telemetry}
