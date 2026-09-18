# DRAC: Diagnosis, Recovery, and Adaptive Control for Fault-Tolerant LLM Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-100%25%20passing-brightgreen.svg)](test_production_suite.py)
[![Paper](https://img.shields.io/badge/paper-full%20draft-orange.svg)](paper/DRAC_FULL_PAPER_DRAFT.md)
[![Venues](https://img.shields.io/badge/target-ASE%20%7C%20FSE%20%7C%20NeurIPS%20%7C%20ICLR-purple.svg)](paper/)

DRAC is a decoupled, closed-loop resilience framework designed to provide autonomous fault diagnosis, transactional state rollback, and budget-constrained recovery for Large Language Model (LLM) agents operating under runtime execution perturbations.

---

## Abstract

Autonomous LLM agents are increasingly deployed in mission-critical environments spanning software engineering, relational database management, and multi-agent coordination. However, runtime tool invocations fail non-deterministically due to network timeouts, upstream server exceptions, schema breaches, and cache staleness. Existing recovery paradigms suffer from a fundamental trilemma:

1. **Context Contamination (Naive Retry):** Appending raw stack traces into the context primes autoregressive transformers to sample tokens from the error-discourse subspace, triggering repeated failures.
2. **Rollback Amnesia (Pure Rollback):** Rewinding context to a previous checkpoint without constraints restores the identical distribution mode, causing greedy samplers to reproduce the exact same flawed action.
3. **Monolithic Self-Diagnosis (Reflexion):** Asking the failing agent in-band to verbally reflect on its own errors incurs runaway token overhead and confabulation without external ground-truth supervision.

DRAC resolves this trilemma by decoupling diagnosis from execution, utilizing transactional state rollback, synthesizing distilled negative constraints (DNCS), and arbitrating recovery actions over a Budget-Constrained Partially Observable Markov Decision Process (B-POMDP).

---

## Architectural Overview

### 1. The LLM Agent Failure Recovery Trilemma

The figure below illustrates the three failure corners of existing recovery techniques and the central Pareto-optimal solution achieved by DRAC through Distilled Negative Constraint Synthesis (DNCS):

<p align="center">
  <img src="paper/figures/recovery_trilemma_diagram.jpg" alt="The Recovery Trilemma" width="850">
</p>

### 2. Closed-Loop B-POMDP Control Architecture

DRAC operates via an out-of-band control loop spanning five distinct stages:

<p align="center">
  <img src="paper/figures/bpomdp_control_loop_diagram.jpg" alt="DRAC B-POMDP Control Loop Architecture" width="850">
</p>

1. **Runtime Telemetry Monitor:** Passively intercepts HTTP status codes, latency spikes, and exceptions without modifying the underlying agent code.
2. **Decoupled Dual-Process Diagnoser:** Bifurcates diagnosis into a deterministic fast-path ($0.010\text{ ms}$, $0\text{ token cost}$, $98.2\%$ accuracy) and a lightweight semantic micro-evaluator.
3. **Budget-Constrained Utility Arbiter (B-POMDP):** Evaluates expected utility over remaining execution token ($C_{\text{rem}}$) and latency ($T_{\text{rem}}$) budgets, enforcing safe human escalation if thresholds are breached.
4. **Transactional State Manager with DNCS:** Prunes corrupted trajectory tokens upon rollback and injects an ultra-compact ($\le 25\text{ token}$) counterfactual constraint into clean checkpoint $S_k$.
5. **Pre-Flight Invariant Verifier:** Validates post-conditions (SQLite return rows, JSON schema integrity) before resuming autonomous execution.

---

## Empirical Benchmark Documentation

Complete experimental results, granular fault-by-fault breakdowns, task workload matrices, and formal mathematical proofs are compiled in dedicated technical reports:

* **[COMPLETE_TABLES_AND_PROOF.md](COMPLETE_TABLES_AND_PROOF.md):** Contains all empirical tables (Tables 1 through 6), formal proofs for Theorems 1–3, and the failure-remediation impact matrix.
* **[paper/DRAC_FULL_PAPER_DRAFT.md](paper/DRAC_FULL_PAPER_DRAFT.md):** The complete 8-section conference paper draft.
* **[experiments/results/raw_trials.csv](experiments/results/raw_trials.csv):** Raw trial-by-trial logs across 400 real-execution evaluations.
* **[experiments/results/summary_metrics.csv](experiments/results/summary_metrics.csv):** Aggregated metrics across all 5 recovery paradigms.

---

## Quick Start

### Installation

```bash
git clone https://github.com/<username>/drac.git
cd drac
pip install -r requirements.txt  # pandas numpy matplotlib tabulate
```

### Verification & Reproduction Commands

Run the 13-stage comprehensive production test suite (100% pass rate):
```bash
python main.py --mode verify
```

Display all publication tables directly in your terminal:
```bash
python main.py --mode tables
```

Execute the full 400-trial benchmark suite and regenerate all figures:
```bash
python main.py --mode all
```

---

## Repository Structure

```
drac/
├── drac/                       # Core DRAC Resilience Engine
│   ├── types.py                # TelemetryEvent, FaultDomain, Checkpoint, Budget
│   ├── detector.py             # Out-of-band invariant anomaly detector
│   ├── diagnoser.py            # Dual-Process Diagnoser (System 1 & System 2)
│   ├── dncs.py                 # Distilled Negative Constraint Synthesizer
│   ├── state_manager.py        # Transactional checkpointer & context pruner
│   ├── arbiter.py              # B-POMDP budget-constrained utility arbiter
│   └── verifier.py             # Pre-flight state & schema invariant verifier
├── agents/                     # Benchmark Agent Architectures
│   ├── calculator_agent.py     # Arithmetic & numerical math tool agent
│   ├── search_agent.py         # Knowledge-retrieval search agent
│   ├── db_agent.py             # Relational in-memory SQLite database agent
│   └── multi_agent_pipeline.py # 4-node collaborative pipeline (Planner->Researcher->Analyst->Reviewer)
├── injector/                   # Chaos Fault Injection Framework
│   ├── fault_types.py          # 16 fine-grained fault definitions across 6 domains
│   └── proxy.py                # Non-invasive runtime tool interception proxy
├── baselines/                  # Comparative Recovery Paradigms
│   └── strategies.py           # Naive Retry, Reflexion, Pure Rollback, DRAC Fixed, DRAC Full
├── experiments/                # Empirical Benchmarking Suite
│   ├── metrics.py              # SRE Metrics engine (RSR, RCA, FDR, TOR, CCF, CNRE)
│   ├── run_benchmarks.py       # 400-trial real execution benchmark runner
│   ├── plot_results.py         # Figures 1 to 4 publication plotter
│   ├── plot_theoretical_diagrams.py # Figures 5 & 6 300 DPI vector plot generator
│   └── plots/                  # Generated high-resolution publication figures
├── paper/                      # Academic Manuscript & BibTeX Database
│   ├── DRAC_FULL_PAPER_DRAFT.md# Complete 8-section research paper draft
│   ├── COMPLETE_TABLES_AND_PROOF.md # Dedicated empirical tables and mathematical proofs
│   ├── references.bib          # 45-paper BibTeX bibliography (2023–2026)
│   └── figures/                # High-res diagrams & vector plots
├── main.py                     # Master CLI runner (--mode all|verify|benchmark|tables)
├── test_production_suite.py    # Master 9-test production verification suite
├── LICENSE                     # MIT Open Source License
└── README.md
```

---

## Citation

```bibtex
@article{drac2026faulttolerance,
  title={DRAC: Diagnosis, Recovery, and Adaptive Control for Fault-Tolerant LLM Agents},
  author={Senior Research Team},
  journal={arXiv preprint arXiv:2603.XXXXX},
  year={2026},
  url={https://github.com/<username>/drac}
}
```
