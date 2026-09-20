"""
Benchmark Agent 1: Scientific Calculator & Math Tool Agent.
Performs multi-step arithmetic, unit conversions, and algebraic evaluations.

L1 Fix: Optionally uses GroqLLMClient to decide which expression to evaluate,
proving DRAC is model-agnostic. Set GROQ_API_KEY env var to enable.
Falls back to deterministic Python logic if no key is provided.
"""
from typing import Dict, Any, List, Optional
import math
from injector.proxy import RuntimeFaultProxy
try:
    from drac.llm_client import GroqLLMClient
except ImportError:
    GroqLLMClient = None  # type: ignore

class CalculatorAgent:
    def __init__(self, proxy: RuntimeFaultProxy, llm_client=None):
        self.proxy = proxy
        self.memory: Dict[str, Any] = {}
        self.context: List[Dict[str, Any]] = []
        self.llm = llm_client
        # WHY optional llm_client: when set (GroqLLMClient), the agent asks
        # the LLM what expression to evaluate instead of using the passed value.
        # DRAC still intercepts the tool call via proxy -- proving model-agnosticism.

    def execute_task(self, expression: str, task_prompt: str) -> Dict[str, Any]:
        self.context.append({"role": "user", "content": task_prompt})

        # L1 Fix: if a real LLM is available, ask it what expression to evaluate.
        # WHY: proves DRAC works with non-deterministic LLM decisions, not just
        # hardcoded Python. DRAC monitors the tool call -- not the LLM call.
        actual_expr = expression
        llm_tokens = 0
        if self.llm is not None and getattr(self.llm, 'is_available', False):
            llm_resp = self.llm.ask(
                context=[{"role": "user", "content": task_prompt}],
                task_hint=f"evaluate expression: {expression}"
            )
            if llm_resp.get("source") == "groq" and llm_resp.get("value"):
                actual_expr = llm_resp["value"]
            llm_tokens = llm_resp.get("tokens_used", 0)

        def _calc():
            # Safe deterministic evaluation -- allowed math namespace only
            allowed = {"math": math, "abs": abs, "round": round,
                       "pow": pow, "sqrt": math.sqrt}
            return eval(actual_expr, {"__builtins__": None}, allowed)

        res, telemetry = self.proxy.intercept_tool_call(
            agent_id="calculator_agent",
            tool_name="calculator_eval",
            tool_args={"expression": actual_expr},
            execute_fn=_calc
        )

        if telemetry.raw_error or res is None:
            self.context.append({"role": "assistant", "content": f"Tool error: {telemetry.raw_error}"})
            return {"success": False, "result": None, "telemetry": telemetry}

        self.memory["last_result"] = res
        self.context.append({"role": "assistant", "content": f"Result: {res}"})
        return {"success": True, "result": res, "telemetry": telemetry}
