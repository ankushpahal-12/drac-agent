"""
Benchmark Agent 4: Multi-Agent Collaborative Pipeline (MAS).
Collaborative topology: Planner -> Researcher -> Analyst -> Reviewer.
"""
from typing import Dict, Any, List
from injector.proxy import RuntimeFaultProxy

class MultiAgentPipeline:
    def __init__(self, proxy: RuntimeFaultProxy):
        self.proxy = proxy
        self.messages: List[Dict[str, Any]] = []

    def run_pipeline(self, user_prompt: str) -> Dict[str, Any]:
        self.messages.clear()
        
        # Step 1: Planner
        plan_msg = {"from": "Planner", "to": "Researcher", "content": f"Subtasks for: {user_prompt}"}
        
        # Step 2: Researcher (Intercepted)
        def _research():
            return {"status": "ok", "findings": ["Metric A is 92%", "Metric B is 88%"]}

        res_findings, telemetry_res = self.proxy.intercept_tool_call(
            agent_id="researcher_agent",
            tool_name="inter_agent_send",
            tool_args=plan_msg,
            execute_fn=_research
        )
        if telemetry_res.raw_error or res_findings is None:
            return {
                "success": False,
                "failed_agent": "researcher_agent",
                "telemetry": telemetry_res,
                "cascade_contained": True  # If stopped here
            }

        # Step 3: Analyst
        analyst_out = {"analysis": "Metrics indicate high reliability", "score": 0.90}

        # Step 4: Reviewer
        final_report = {"verified": True, "summary": analyst_out["analysis"]}
        return {
            "success": True,
            "failed_agent": None,
            "telemetry": telemetry_res,
            "final_report": final_report,
            "cascade_contained": True
        }
