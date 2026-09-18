"""
Benchmark Agent 2: Web Research & Verification Agent.
Queries simulated search index, retrieves sources, and verifies factual claims.
"""
from typing import Dict, Any, List
from injector.proxy import RuntimeFaultProxy

class SearchAgent:
    def __init__(self, proxy: RuntimeFaultProxy):
        self.proxy = proxy
        self.context: List[Dict[str, Any]] = []
        self.knowledge_base = {
            "drac latency": "DRAC average fast-path latency is 0.001s compared to 1.2s for slow-path.",
            "agentchaos ase": "AgentChaos was accepted at ASE 2026, evaluating 65 fault configurations.",
            "mas-fire topology": "MAS-FIRE demonstrated closed-loop topology neutralizes over 40% of faults."
        }

    def execute_task(self, query: str, task_prompt: str) -> Dict[str, Any]:
        self.context.append({"role": "user", "content": task_prompt})

        def _search():
            q_clean = query.lower().strip()
            for k, v in self.knowledge_base.items():
                if any(word in q_clean for word in k.split()):
                    return {"query": query, "snippets": [v], "total_hits": 1}
            return {"query": query, "snippets": [], "total_hits": 0}

        res, telemetry = self.proxy.intercept_tool_call(
            agent_id="search_agent",
            tool_name="web_search",
            tool_args={"query": query},
            execute_fn=_search
        )

        is_empty_res = (res is None) or (isinstance(res, str) and len(res.strip()) == 0) or (isinstance(res, dict) and res.get("total_hits", 1) == 0)
        if telemetry.raw_error or is_empty_res:
            self.context.append({"role": "assistant", "content": f"Search failed: {telemetry.raw_error or 'Empty search results'}"})
            return {"success": False, "result": None, "telemetry": telemetry}

        self.context.append({"role": "assistant", "content": f"Found: {res}"})
        return {"success": True, "result": res, "telemetry": telemetry}
