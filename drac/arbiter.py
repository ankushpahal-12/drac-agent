"""
DRAC Budget-Constrained Recovery Arbiter (B-POMDP Utility Maximizer).
Selects optimal recovery action conditioned on diagnosis, severity, confidence,
and remaining token/latency budgets.
"""
from typing import Dict, Tuple, Optional
from drac.types import (
    RecoveryAction, DiagnosisResult, FaultDomain, FaultType, Severity, ExecutionBudget
)

class RecoveryArbiter:
    def __init__(self, lambda_cost: float = 0.25, lambda_latency: float = 0.15):
        self.lambda_cost = lambda_cost
        self.lambda_latency = lambda_latency

        # Empirical baseline success probabilities P(Success | Domain, Severity, Action)
        self.prior_success_matrix: Dict[Tuple[FaultDomain, RecoveryAction], float] = {
            (FaultDomain.TOOL, RecoveryAction.RETRY): 0.20,
            (FaultDomain.TOOL, RecoveryAction.ROLLBACK_WITH_DNCS): 0.92,
            (FaultDomain.TOOL, RecoveryAction.PURE_ROLLBACK): 0.35,  # Amnesia penalty
            (FaultDomain.TOOL, RecoveryAction.FALLBACK_TOOL): 0.88,
            (FaultDomain.TOOL, RecoveryAction.REPLAN): 0.65,

            (FaultDomain.OUTPUT_SCHEMA, RecoveryAction.RETRY): 0.40,
            (FaultDomain.OUTPUT_SCHEMA, RecoveryAction.ROLLBACK_WITH_DNCS): 0.94,
            (FaultDomain.OUTPUT_SCHEMA, RecoveryAction.PURE_ROLLBACK): 0.42,
            (FaultDomain.OUTPUT_SCHEMA, RecoveryAction.REPLAN): 0.70,

            (FaultDomain.PLANNING, RecoveryAction.RETRY): 0.10,
            (FaultDomain.PLANNING, RecoveryAction.REPLAN): 0.85,
            (FaultDomain.PLANNING, RecoveryAction.ROLLBACK_WITH_DNCS): 0.89,
            (FaultDomain.PLANNING, RecoveryAction.PURE_ROLLBACK): 0.25,

            (FaultDomain.CONTEXT, RecoveryAction.RETRY): 0.05,
            (FaultDomain.CONTEXT, RecoveryAction.ROLLBACK_WITH_DNCS): 0.95,
            (FaultDomain.CONTEXT, RecoveryAction.PURE_ROLLBACK): 0.30,
            (FaultDomain.CONTEXT, RecoveryAction.REPLAN): 0.50,

            (FaultDomain.COMMUNICATION, RecoveryAction.RETRY): 0.35,
            (FaultDomain.COMMUNICATION, RecoveryAction.REPLAN): 0.75,
            (FaultDomain.COMMUNICATION, RecoveryAction.ROLLBACK_WITH_DNCS): 0.90,
            (FaultDomain.COMMUNICATION, RecoveryAction.FALLBACK_TOOL): 0.80,
        }

        # Expected resource expenditures by action
        self.action_cost_estimates: Dict[RecoveryAction, Tuple[int, float]] = {
            RecoveryAction.RETRY: (150, 0.5),                # (tokens, seconds)
            RecoveryAction.REPLAN: (450, 1.2),
            RecoveryAction.ROLLBACK_WITH_DNCS: (180, 0.6),   # Compact constraint
            RecoveryAction.PURE_ROLLBACK: (160, 0.5),
            RecoveryAction.FALLBACK_TOOL: (220, 0.7),
            RecoveryAction.ALTERNATE_MODEL: (600, 2.0),
            RecoveryAction.HUMAN_ESCALATION: (0, 0.0),
        }

    def select_action(self, diagnosis: DiagnosisResult, budget: ExecutionBudget, consecutive_failures: int = 0) -> Tuple[RecoveryAction, float]:
        """
        Evaluate utility U(a) for each candidate action under remaining budget.
        Returns (optimal_action, max_utility).
        """
        # If budget exhausted or repeated failures exceed safety limit -> escalate
        if consecutive_failures >= 3 or budget.tokens_remaining < 100 or budget.time_remaining < 1.0:
            return RecoveryAction.HUMAN_ESCALATION, 0.0

        candidate_actions = [
            RecoveryAction.RETRY,
            RecoveryAction.REPLAN,
            RecoveryAction.ROLLBACK_WITH_DNCS,
            RecoveryAction.FALLBACK_TOOL,
            RecoveryAction.ALTERNATE_MODEL
        ]

        best_action = RecoveryAction.RETRY
        best_utility = -float("inf")

        for action in candidate_actions:
            est_tokens, est_lat = self.action_cost_estimates.get(action, (200, 1.0))
            if est_tokens > budget.tokens_remaining or est_lat > budget.time_remaining:
                continue  # Infeasible under strict budget

            p_succ = self.prior_success_matrix.get((diagnosis.domain, action), 0.50)
            
            # Penalize simple retry under high severity or persistent errors
            if action == RecoveryAction.RETRY and diagnosis.severity == Severity.HIGH:
                p_succ *= 0.3

            # Calculate normalized cost and latency penalty
            cost_penalty = self.lambda_cost * (est_tokens / max(100, budget.tokens_remaining))
            lat_penalty = self.lambda_latency * (est_lat / max(1.0, budget.time_remaining))

            utility = (diagnosis.confidence * p_succ) - cost_penalty - lat_penalty

            if utility > best_utility:
                best_utility = utility
                best_action = action

        if best_utility == -float("inf"):
            return RecoveryAction.HUMAN_ESCALATION, 0.0

        return best_action, best_utility
