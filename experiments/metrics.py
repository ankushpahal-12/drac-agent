"""
DRAC Empirical Metrics & Statistical Evaluation Engine.
Computes FDR, RCA, RSR, RL, RC, CNRE, CCF, TOR, MTTR_A, and MTCR_A.
"""
import math
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

@dataclass
class TrialResult:
    task_name: str
    fault_type: str
    ground_truth_domain: str
    strategy_name: str
    fault_detected: bool
    diagnosed_domain: Optional[str]
    diagnosis_correct: bool
    recovery_success: bool
    recovery_latency_sec: float
    recovery_cost_tokens: int
    base_task_tokens: int = 400
    cascade_contained: bool = True
    action_taken: Optional[str] = None

class MetricEvaluator:
    def __init__(self, trials: List[TrialResult]):
        self.trials = trials
        self.df = pd.DataFrame([t.__dict__ for t in trials])

    def compute_summary_by_strategy(self) -> pd.DataFrame:
        """
        Compute consolidated metrics across all recovery strategies.
        """
        # Baseline reference for normalization (Naive Retry)
        naive_df = self.df[self.df["strategy_name"] == "Naive Retry"]
        base_rc = naive_df["recovery_cost_tokens"].mean() if not naive_df.empty else 150.0
        base_rl = naive_df["recovery_latency_sec"].mean() if not naive_df.empty else 0.5

        records = []
        for strat, grp in self.df.groupby("strategy_name"):
            n_total = len(grp)
            n_detected = grp["fault_detected"].sum()
            n_diag_correct = grp["diagnosis_correct"].sum()
            n_recovered = grp["recovery_success"].sum()

            fdr = (n_detected / n_total) * 100.0 if n_total > 0 else 0.0
            rca = (n_diag_correct / n_detected) * 100.0 if n_detected > 0 else 0.0
            rsr = (n_recovered / n_total) * 100.0 if n_total > 0 else 0.0
            rl_mean = grp["recovery_latency_sec"].mean()
            rc_mean = grp["recovery_cost_tokens"].mean()
            
            # Successful subset for MTTR / MTCR
            succ_grp = grp[grp["recovery_success"] == True]
            mttr_a = succ_grp["recovery_latency_sec"].mean() if not succ_grp.empty else float("nan")
            mtcr_a = succ_grp["recovery_cost_tokens"].mean() if not succ_grp.empty else float("nan")

            # CCF (Cascade Containment Factor)
            ccf = (grp["cascade_contained"].sum() / n_total) * 100.0 if n_total > 0 else 100.0

            # TOR (Token Overhead Ratio)
            tor = (rc_mean / grp["base_task_tokens"].mean()) * 100.0

            # CNRE Formulation
            norm_cost = rc_mean / max(1.0, base_rc)
            norm_lat = rl_mean / max(0.01, base_rl)
            denom = math.log2(1.0 + norm_cost) * math.log2(1.0 + norm_lat)
            cnre = (rsr / 100.0) / max(0.05, denom)

            records.append({
                "Strategy": strat,
                "FDR (%)": round(fdr, 1),
                "RCA (%)": round(rca, 1),
                "RSR (%)": round(rsr, 1),
                "RL (s)": round(rl_mean, 3),
                "RC (tokens)": round(rc_mean, 1),
                "MTTR_A (s)": round(mttr_a, 3),
                "MTCR_A (tok)": round(mtcr_a, 1),
                "CCF (%)": round(ccf, 1),
                "TOR (%)": round(tor, 1),
                "CNRE": round(cnre, 3)
            })

        summary_df = pd.DataFrame(records)
        # Sort by CNRE descending
        return summary_df.sort_values(by="RSR (%)", ascending=False)

    def generate_markdown_table(self) -> str:
        df = self.compute_summary_by_strategy()
        return df.to_markdown(index=False)
