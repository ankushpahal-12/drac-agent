"""
Benchmark Agent 4: Multi-Agent Collaborative Pipeline (MAS).
Collaborative topology: Planner -> Researcher -> Analyst -> Reviewer.

The pipeline uses real computation at every stage:
- Researcher: real SQLite FTS5 BM25 search over the research corpus
- Analyst: computes a real reliability score from the retrieved BM25 scores
- Reviewer: validates that the score is within a physically plausible range [0, 1]
"""
import math
from typing import Dict, Any, List
from injector.proxy import RuntimeFaultProxy
from agents.search_agent import FTSSearchIndex


class MultiAgentPipeline:
    def __init__(self, proxy: RuntimeFaultProxy):
        self.proxy = proxy
        self.messages: List[Dict[str, Any]] = []
        # Shared FTS5 index — built once, reused across pipeline steps
        self._index = FTSSearchIndex()

    def run_pipeline(self, user_prompt: str) -> Dict[str, Any]:
        self.messages.clear()

        # ---------------------------------------------------------------
        # Step 1: Planner — decomposes the user prompt into subtasks
        # ---------------------------------------------------------------
        plan_msg = {
            "from": "Planner",
            "to": "Researcher",
            "content": f"Find recovery metrics and reliability statistics for: {user_prompt}"
        }
        self.messages.append(plan_msg)

        # ---------------------------------------------------------------
        # Step 2: Researcher — runs real BM25 FTS5 search (intercepted)
        # ---------------------------------------------------------------
        def _research():
            """
            Real BM25 full-text search using the FTS5 index.
            Findings are the actual top-ranked snippets with real relevance scores.
            """
            results = self._index.search("fault recovery rollback agent reliability", top_k=4)
            if not results:
                results = self._index.search("agent failure", top_k=4)
            if not results:
                return None  # No findings — propagates as failure

            # Real findings extracted from actual corpus matches
            findings = [
                f"{r['title']} (BM25={r['bm25_score']:.3f}): {r['snippet']}"
                for r in results
            ]
            return {
                "status": "ok",
                "findings": findings,
                "raw_results": results        # carry raw results for analyst
            }

        res_findings, telemetry_res = self.proxy.intercept_tool_call(
            agent_id="researcher_agent",
            tool_name="fts_search_pipeline",
            tool_args=plan_msg,
            execute_fn=_research
        )

        if telemetry_res.raw_error or res_findings is None or not isinstance(res_findings, dict):
            return {
                "success": False,
                "failed_agent": "researcher_agent",
                "telemetry": telemetry_res,
                "cascade_contained": True
            }

        # ---------------------------------------------------------------
        # Step 3: Analyst — computes real reliability score
        # ---------------------------------------------------------------
        raw_results = res_findings.get("raw_results", [])

        if raw_results:
            # Real computation: mean BM25 score of top results
            mean_bm25 = sum(r["bm25_score"] for r in raw_results) / len(raw_results)
            # Logistic normalization: maps BM25 ∈ [0, ∞) → score ∈ (0, 1)
            # Formula: σ(x) = 1 / (1 + e^(-k*(x - x0))) with k=0.3, x0=5.0
            #
            # WHY k=0.3 (logistic slope):
            #   At k=0.3 and x0=5.0: BM25=5 → score=0.50, BM25=10 → score=0.82, BM25=2 → score=0.27.
            #   This maps the FTS5 BM25 range [0, ~15] onto [0.1, 0.95] — physiologically plausible
            #   reliability scores. Larger k would make the function too step-like; smaller k too flat.
            # WHERE: Determines the reliability_score output of the Analyst stage.
            #
            # WHY x0=5.0 (logistic midpoint = inflection point):
            #   BM25 ≈ 5.0 is the "moderate relevance" threshold in FTS5 (documents scoring above 5
            #   are considered clearly on-topic). Setting x0=5 means moderate match → 50% reliability,
            #   strong match → > 80%, weak match (BM25 < 2) → < 27%.
            # WHERE: Center of the reliability_score mapping curve.
            reliability_score = round(1.0 / (1.0 + math.exp(-0.3 * (mean_bm25 - 5.0))), 4)
        else:
            reliability_score = 0.0

        analyst_out = {
            "analysis": f"Recovery reliability derived from {len(raw_results)} corpus matches",
            "reliability_score": reliability_score,      # real computed value in [0,1]
            "mean_bm25": round(mean_bm25 if raw_results else 0.0, 4),
            "num_sources": len(raw_results)
        }

        # ---------------------------------------------------------------
        # Step 4: Reviewer — validates the computed score is physically valid
        # ---------------------------------------------------------------
        score_valid = 0.0 < analyst_out["reliability_score"] <= 1.0
        final_report = {
            "verified": score_valid,
            "reliability_score": analyst_out["reliability_score"],
            "mean_bm25": analyst_out["mean_bm25"],
            "summary": analyst_out["analysis"],
            "num_sources": analyst_out["num_sources"]
        }

        return {
            "success": True,
            "failed_agent": None,
            "telemetry": telemetry_res,
            "final_report": final_report,
            "cascade_contained": True
        }
