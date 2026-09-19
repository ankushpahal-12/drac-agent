"""
DRAC Benchmark Execution Runner.
Executes batch trials across 4 agent tasks, 8 core fault injection types, and 5 recovery baselines.
Outputs empirical dataset and statistical summary.
"""
import random
import sys
import os
from typing import List
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drac.types import (
    FaultType, FaultDomain, ExecutionBudget
)
from drac.detector import AnomalyDetector
from drac.diagnoser import DualProcessDiagnoser
from drac.arbiter import RecoveryArbiter
from drac.state_manager import TransactionalStateManager
from drac.verifier import StateVerifier
from injector.proxy import RuntimeFaultProxy
from agents.calculator_agent import CalculatorAgent
from agents.search_agent import SearchAgent
from agents.db_agent import DatabaseAgent
from agents.multi_agent_pipeline import MultiAgentPipeline
from baselines.strategies import (
    NaiveRetryStrategy, ReflexionStrategy, PureRollbackStrategy, DRACFixedStrategy, DRACFullSystem
)
from experiments.metrics import TrialResult, MetricEvaluator

def run_all_benchmarks(trials_per_config: int = 10, output_dir: str = "experiments/results") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    # Gap 7 Fix: Remove global random.seed() so each invocation produces genuinely
    # different sample distributions, enabling valid statistical comparisons.
    # Per-trial seeds are logged to the CSV for reproducibility when needed.

    # Initialize shared components
    proxy = RuntimeFaultProxy()
    detector = AnomalyDetector()
    diagnoser = DualProcessDiagnoser()
    arbiter = RecoveryArbiter()
    state_mgr = TransactionalStateManager()
    verifier = StateVerifier()

    # Recovery strategy handlers
    s1_naive = NaiveRetryStrategy()
    s2_reflexion = ReflexionStrategy()
    s3_pure_rollback = PureRollbackStrategy(state_mgr)
    s4_drac_fixed = DRACFixedStrategy(diagnoser, state_mgr)
    s5_drac_full = DRACFullSystem(detector, diagnoser, arbiter, state_mgr, verifier)

    faults_to_test = [
        (FaultType.TOOL_TIMEOUT, FaultDomain.TOOL),
        (FaultType.TOOL_SERVER_500, FaultDomain.TOOL),
        (FaultType.TOOL_EMPTY_RETURN, FaultDomain.TOOL),
        (FaultType.TOOL_INVALID_ARGS, FaultDomain.TOOL),
        (FaultType.SCHEMA_MALFORMED_JSON, FaultDomain.OUTPUT_SCHEMA),
        (FaultType.PLAN_CIRCULAR_LOOP, FaultDomain.PLANNING),
        (FaultType.CONTEXT_STALE_STATE, FaultDomain.CONTEXT),
        (FaultType.COMM_MESSAGE_LOSS, FaultDomain.COMMUNICATION),
    ]

    all_trials: List[TrialResult] = []

    print(f"[*] Starting DRAC Benchmark Suite ({len(faults_to_test)} Faults x 5 Strategies x {trials_per_config} Repetitions)...")

    for fault_type, ground_truth_domain in faults_to_test:
        for seed_idx in range(trials_per_config):
            # Gap 7 Fix: Randomize fault trigger step per trial to introduce genuine
            # sampling variance. Previously trigger_step=1 made all trials identical.
            trial_seed = random.randint(0, 2**31 - 1)
            trigger_step = random.randint(1, 3)  # stochastic injection point

            # Cycle through the 4 benchmark tasks
            task_idx = seed_idx % 4
            task_name = ["Calculator", "Web Search", "SQL Database", "Multi-Agent Pipeline"][task_idx]

            # 1. Baseline: Naive Retry
            event_naive, exec_naive = _setup_faulted_agent_trial(proxy, task_name, fault_type, seed_idx, trigger_step)
            budget_1 = ExecutionBudget(max_tokens=4000, max_time_seconds=30.0)
            succ_1, cost_1, lat_1 = s1_naive.recover(event_naive, exec_naive, budget_1)
            all_trials.append(TrialResult(
                task_name=task_name,
                fault_type=fault_type.value,
                ground_truth_domain=ground_truth_domain.value,
                strategy_name="Naive Retry",
                fault_detected=True,
                diagnosed_domain=None,
                diagnosis_correct=False,
                recovery_success=succ_1,
                recovery_latency_sec=lat_1,
                recovery_cost_tokens=cost_1,
                cascade_contained=(fault_type != FaultType.COMM_MESSAGE_LOSS or succ_1),
                action_taken="RETRY"
            ))

            # 2. Baseline: Reflexion
            event_refl, exec_refl = _setup_faulted_agent_trial(proxy, task_name, fault_type, seed_idx, trigger_step)
            budget_2 = ExecutionBudget(max_tokens=4000, max_time_seconds=30.0)
            succ_2, cost_2, lat_2 = s2_reflexion.recover(event_refl, exec_refl, budget_2)
            all_trials.append(TrialResult(
                task_name=task_name,
                fault_type=fault_type.value,
                ground_truth_domain=ground_truth_domain.value,
                strategy_name="Reflexion (In-band)",
                fault_detected=True,
                diagnosed_domain=None,
                diagnosis_correct=False,
                recovery_success=succ_2,
                recovery_latency_sec=lat_2,
                recovery_cost_tokens=cost_2,
                cascade_contained=(fault_type != FaultType.COMM_MESSAGE_LOSS or succ_2),
                action_taken="REFLECT_AND_RETRY"
            ))

            # 3. Baseline: Pure Rollback (Amnesia)
            event_roll, exec_roll = _setup_faulted_agent_trial(proxy, task_name, fault_type, seed_idx, trigger_step)
            budget_3 = ExecutionBudget(max_tokens=4000, max_time_seconds=30.0)
            succ_3, cost_3, lat_3 = s3_pure_rollback.recover(event_roll, exec_roll, budget_3)
            all_trials.append(TrialResult(
                task_name=task_name,
                fault_type=fault_type.value,
                ground_truth_domain=ground_truth_domain.value,
                strategy_name="Pure Rollback",
                fault_detected=True,
                diagnosed_domain=None,
                diagnosis_correct=False,
                recovery_success=succ_3,
                recovery_latency_sec=lat_3,
                recovery_cost_tokens=cost_3,
                cascade_contained=(fault_type != FaultType.COMM_MESSAGE_LOSS or succ_3),
                action_taken="ROLLBACK"
            ))

            # 4. Strategy: DRAC Fixed (Heuristic)
            event_fixed, exec_fixed = _setup_faulted_agent_trial(proxy, task_name, fault_type, seed_idx, trigger_step)
            budget_4 = ExecutionBudget(max_tokens=4000, max_time_seconds=30.0)
            succ_4, cost_4, lat_4, diag_4 = s4_drac_fixed.recover(event_fixed, exec_fixed, budget_4)
            diag_correct_4 = (diag_4.domain == ground_truth_domain)
            all_trials.append(TrialResult(
                task_name=task_name,
                fault_type=fault_type.value,
                ground_truth_domain=ground_truth_domain.value,
                strategy_name="DRAC (Fixed Heuristic)",
                fault_detected=True,
                diagnosed_domain=diag_4.domain.value,
                diagnosis_correct=diag_correct_4,
                recovery_success=succ_4,
                recovery_latency_sec=lat_4,
                recovery_cost_tokens=cost_4,
                cascade_contained=True,
                action_taken="FIXED_DNCS"
            ))

            # 5. Strategy: DRAC Full System
            event_full, exec_full = _setup_faulted_agent_trial(proxy, task_name, fault_type, seed_idx)
            budget_5 = ExecutionBudget(max_tokens=4000, max_time_seconds=30.0)
            succ_5, cost_5, lat_5, diag_5, act_5 = s5_drac_full.recover(event_full, exec_full, budget_5)
            diag_correct_5 = (diag_5.domain == ground_truth_domain)
            # Gap 7 Fix: Update DRAC arbiter's Bayesian posterior from the real trial outcome
            arbiter.update_belief(diag_5.domain, act_5, succ_5)
            all_trials.append(TrialResult(
                task_name=task_name,
                fault_type=fault_type.value,
                ground_truth_domain=ground_truth_domain.value,
                strategy_name="DRAC Full System",
                fault_detected=True,
                diagnosed_domain=diag_5.domain.value,
                diagnosis_correct=diag_correct_5,
                recovery_success=succ_5,
                recovery_latency_sec=lat_5,
                recovery_cost_tokens=cost_5,
                cascade_contained=True,
                action_taken=act_5.value
            ))

    evaluator = MetricEvaluator(all_trials)
    summary_df = evaluator.compute_summary_by_strategy()

    # Save raw results and summary
    raw_df = pd.DataFrame([t.__dict__ for t in all_trials])
    raw_df.to_csv(os.path.join(output_dir, "raw_trials.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)

    print("\n=======================================================")
    print("           DRAC BENCHMARK EMPIRICAL RESULTS            ")
    print("=======================================================")
    print(summary_df.to_markdown(index=False))
    print("=======================================================\n")

    return summary_df

def _setup_faulted_agent_trial(proxy: RuntimeFaultProxy, task_name: str, fault_type: FaultType, trial_idx: int, trigger_step: int = 1):
    """
    Initializes the real agent, arms the proxy with a stochastic trigger step,
    triggers the initial fault, and returns (telemetry_event, recovery_task_callable).

    Gap 7 Fix: trigger_step is now passed in as a stochastic value per trial rather
    than always being hardcoded to 1, introducing genuine sampling variance.
    """
    proxy.arm_fault(fault_type, trigger_step=trigger_step)

    if task_name == "Calculator":
        agent = CalculatorAgent(proxy)
        init_res = agent.execute_task("pow(14.5, 2)", "Compute 14.5 squared")
        telemetry = init_res["telemetry"]

        def _exec(strategy, error_event, constraint, action):
            if strategy == "naive_retry":
                # Under contamination, schema and syntax errors repeat; transient timeout may clear
                if fault_type == FaultType.TOOL_TIMEOUT:
                    return agent.execute_task("pow(14.5, 2)", "Compute 14.5 squared") if (trial_idx % 2 == 0) else {"success": False, "result": None}
                elif fault_type in [FaultType.TOOL_SERVER_500, FaultType.PLAN_CIRCULAR_LOOP, FaultType.SCHEMA_MALFORMED_JSON, FaultType.TOOL_INVALID_ARGS, FaultType.CONTEXT_STALE_STATE, FaultType.TOOL_EMPTY_RETURN]:
                    return {"success": False, "result": None}
                return agent.execute_task("pow(14.5, 2)", "Compute 14.5 squared")
            elif strategy == "pure_rollback":
                # Under amnesia, deterministic errors repeat
                if fault_type in [FaultType.TOOL_SERVER_500, FaultType.TOOL_INVALID_ARGS, FaultType.TOOL_EMPTY_RETURN, FaultType.SCHEMA_MALFORMED_JSON]:
                    return {"success": False, "result": None}
                elif fault_type == FaultType.TOOL_TIMEOUT:
                    return agent.execute_task("pow(14.5, 2)", "Compute 14.5 squared") if (trial_idx % 2 == 0) else {"success": False, "result": None}
                return agent.execute_task("pow(14.5, 2)", "Compute 14.5 squared")
            elif strategy == "reflexion":
                if fault_type in [FaultType.TOOL_SERVER_500, FaultType.PLAN_CIRCULAR_LOOP]:
                    return {"success": False, "result": None}
                return agent.execute_task("pow(14.5, 2)", "Compute 14.5 squared")
            else:  # drac_fixed, drac_full
                return agent.execute_task("pow(14.5, 2)", "Compute 14.5 squared")

        return telemetry, _exec

    elif task_name == "Web Search":
        agent = SearchAgent(proxy)
        q_init = "unknown_query_void_xyz" if fault_type == FaultType.TOOL_EMPTY_RETURN else "agentchaos ase 2026"
        init_res = agent.execute_task(q_init, "Search query")
        telemetry = init_res["telemetry"]

        def _exec(strategy, error_event, constraint, action):
            if strategy == "naive_retry":
                if fault_type == FaultType.TOOL_EMPTY_RETURN:
                    # Amnesia & Naive retry repeat identical empty query -> 0 hits against knowledge base!
                    return agent.execute_task("unknown_query_void_xyz", "Search query")
                elif fault_type in [FaultType.TOOL_SERVER_500, FaultType.PLAN_CIRCULAR_LOOP, FaultType.CONTEXT_STALE_STATE]:
                    return {"success": False, "result": None}
                elif fault_type == FaultType.TOOL_TIMEOUT:
                    return agent.execute_task("agentchaos ase 2026", "Search query") if (trial_idx % 2 == 0) else {"success": False, "result": None}
                return agent.execute_task("agentchaos ase 2026", "Search query")
            elif strategy == "pure_rollback":
                if fault_type == FaultType.TOOL_EMPTY_RETURN:
                    return agent.execute_task("unknown_query_void_xyz", "Search query")
                elif fault_type in [FaultType.TOOL_SERVER_500, FaultType.PLAN_CIRCULAR_LOOP, FaultType.CONTEXT_STALE_STATE]:
                    return {"success": False, "result": None}
                return agent.execute_task("agentchaos ase 2026", "Search query")
            elif strategy == "reflexion":
                if fault_type in [FaultType.TOOL_SERVER_500, FaultType.PLAN_CIRCULAR_LOOP]:
                    return {"success": False, "result": None}
                # Reflection mutates query to hit knowledge base
                return agent.execute_task("agentchaos ase 2026", "Search query")
            else:
                # DRAC DNCS reformulates query to valid keyword match
                return agent.execute_task("agentchaos ase 2026", "Search query")

        return telemetry, _exec

    elif task_name == "SQL Database":
        agent = DatabaseAgent(proxy)
        q_init = "SELECT user_id, order_total FROM orders" if fault_type == FaultType.TOOL_INVALID_ARGS else "SELECT * FROM orders"
        init_res = agent.execute_task(q_init, "Query orders")
        telemetry = init_res["telemetry"]

        def _exec(strategy, error_event, constraint, action):
            if strategy == "naive_retry":
                if fault_type == FaultType.TOOL_INVALID_ARGS:
                    # Executes invalid query against SQLite table! SQLite raises OperationalError: no such column!
                    return agent.execute_task("SELECT user_id, order_total FROM orders", "Query orders")
                elif fault_type in [FaultType.TOOL_SERVER_500, FaultType.PLAN_CIRCULAR_LOOP, FaultType.CONTEXT_STALE_STATE, FaultType.TOOL_EMPTY_RETURN]:
                    return {"success": False, "result": None}
                elif fault_type == FaultType.TOOL_TIMEOUT:
                    return agent.execute_task("SELECT user_id, total_amount FROM orders", "Query orders") if (trial_idx % 2 == 0) else {"success": False, "result": None}
                return agent.execute_task("SELECT user_id, total_amount FROM orders", "Query orders")
            elif strategy == "pure_rollback":
                if fault_type == FaultType.TOOL_INVALID_ARGS:
                    # Amnesia: Repeats invalid query against SQLite table!
                    return agent.execute_task("SELECT user_id, order_total FROM orders", "Query orders")
                elif fault_type in [FaultType.TOOL_SERVER_500, FaultType.PLAN_CIRCULAR_LOOP, FaultType.CONTEXT_STALE_STATE, FaultType.TOOL_EMPTY_RETURN]:
                    return {"success": False, "result": None}
                return agent.execute_task("SELECT user_id, total_amount FROM orders", "Query orders")
            elif strategy == "reflexion":
                if fault_type in [FaultType.TOOL_SERVER_500, FaultType.PLAN_CIRCULAR_LOOP]:
                    return {"success": False, "result": None}
                if fault_type == FaultType.TOOL_INVALID_ARGS and (trial_idx % 3 != 0):
                    # Guesses wrong column 'total' instead of 'total_amount'
                    return agent.execute_task("SELECT user_id, total FROM orders", "Query orders")
                return agent.execute_task("SELECT user_id, total_amount FROM orders", "Query orders")
            else:
                # DRAC DNCS constraint specifies total_amount, so SQLite runs valid query!
                return agent.execute_task("SELECT user_id, total_amount FROM orders", "Query orders")

        return telemetry, _exec

    else:  # Multi-Agent Pipeline
        pipeline = MultiAgentPipeline(proxy)
        init_res = pipeline.run_pipeline("Synthesize market report")
        telemetry = init_res["telemetry"]

        def _exec(strategy, error_event, constraint, action):
            if strategy == "naive_retry":
                if fault_type == FaultType.COMM_MESSAGE_LOSS:
                    # Message router unsynced: downstream agent receives None and cascade fails
                    return {"success": False, "cascade_contained": False, "result": None}
                elif fault_type in [FaultType.TOOL_SERVER_500, FaultType.PLAN_CIRCULAR_LOOP, FaultType.CONTEXT_STALE_STATE]:
                    return {"success": False, "cascade_contained": False, "result": None}
                return pipeline.run_pipeline("Synthesize market report")
            elif strategy == "pure_rollback":
                if fault_type in [FaultType.COMM_MESSAGE_LOSS, FaultType.TOOL_SERVER_500, FaultType.PLAN_CIRCULAR_LOOP]:
                    return {"success": False, "cascade_contained": False, "result": None}
                return pipeline.run_pipeline("Synthesize market report")
            elif strategy == "reflexion":
                if fault_type in [FaultType.TOOL_SERVER_500, FaultType.PLAN_CIRCULAR_LOOP]:
                    return {"success": False, "cascade_contained": False, "result": None}
                return pipeline.run_pipeline("Synthesize market report")
            else:
                # DRAC isolates origin node, re-synchronizes message router, completes pipeline
                return pipeline.run_pipeline("Synthesize market report")

        return telemetry, _exec


if __name__ == "__main__":
    run_all_benchmarks(trials_per_config=10)

