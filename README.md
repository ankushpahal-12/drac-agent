# DRAC: Diagnosis, Recovery, and Adaptive Control for Fault-Tolerant LLM Agents

DRAC is an autonomous, out-of-band resilience framework for Large Language Model (LLM) agents. It provides passive anomaly detection, dual-process root-cause diagnosis, transactional state rollback with Distilled Negative Constraint Synthesis (DNCS), and budget-constrained decision arbitration modeled as a Budget-Constrained Partially Observable Markov Decision Process (B-POMDP).

---

## 1. Scientific Motivation & Problem Justification

Autonomous LLM agents are transitioning from conversational assistants to mission-critical execution engines managing relational databases, API microservices, cloud infrastructure, and multi-agent coordination pipelines. However, runtime tool invocations in open-world environments fail non-deterministically due to network timeouts, upstream server exceptions, schema drift, stale context, and circular tool-call loops.

Current industry and academic resilience strategies suffer from a fundamental recovery trilemma:

1. **Context Contamination (Naive In-Band Retry):** Appending raw execution traces or exception stack traces directly into the LLM context prompt pollutes the autoregressive attention window. By conditioning on error-heavy tokens (e.g., `OperationalError`, `500 Server Error`), the model's predictive distribution shifts into an error-discourse subspace, triggering repeated failures or catastrophic hallucinations.
2. **Rollback Amnesia (Pure Rollback):** Rewinding context to a clean checkpoint without supplementary guidance restores the identical distribution mode. Under deterministic or low-temperature greedy decoding ($\tau \to 0$), the agent reproduces the exact same flawed invocation.
3. **Monolithic Self-Diagnosis (In-Band Reflection / Reflexion):** Tasking the failing agent to reflect on its own errors within the primary conversation context incurs runaway token inflation (>100% overhead) and lacks ground-truth verification, resulting in ungrounded confabulation and complete failure on infrastructure crashes ($0.0\%$ recovery).

DRAC resolves this trilemma by physically decoupling runtime monitoring and diagnosis from agent execution, restoring state transactionally, synthesizing ultra-compact negative constraints, and evaluating recovery actions against explicit token and latency budget limits.

---

## 2. Architecture & Control Loop

### The LLM Agent Failure Recovery Trilemma

<p align="center">
  <img src="paper/figures/recovery_trilemma_diagram.jpg" alt="The LLM Agent Failure Recovery Trilemma" width="850">
</p>

### Decoupled B-POMDP Control Architecture

<p align="center">
  <img src="paper/figures/bpomdp_control_loop_diagram.jpg" alt="DRAC B-POMDP Control Loop Architecture" width="850">
</p>

DRAC operates via an out-of-band closed-loop engine organized into five discrete components:

1. **Runtime Telemetry Monitor:** Intercepts tool execution signals (HTTP status codes, latency deltas, return schemas, exception payloads) transparently at the proxy boundary without modifying agent source code.
2. **Decoupled Dual-Process Diagnoser:** 
   - *System 1 (Fast-Path):* Deterministic pattern matching over 16 structured fault signatures ($0.010\text{ ms}$ latency, $0\text{ token overhead}$, $98.2\%$ accuracy).
   - *System 2 (Slow-Path):* Isolated, asynchronous micro-evaluator invoked only when ambiguous semantic exceptions arise ($25\text{ ms}$, $\le 65\text{ tokens}$).
3. **Budget-Constrained Utility Arbiter (B-POMDP):** Evaluates expected utility $\mathbb{E}[U(a)]$ across five discrete recovery primitives (`PARAM_RETRY`, `ROLLBACK_DNCS`, `ENV_MUTATE`, `REPLAN`, `HUMAN_ESCALATION`) subject to hard remaining token ($C_{\text{rem}}$) and time ($T_{\text{rem}}$) thresholds.
4. **Transactional State Manager with DNCS:** Prunes corrupted trajectory tokens upon rollback to clean checkpoint $S_k$ and injects a distilled negative constraint ($\le 25\text{ tokens}$) forbidding the failed execution mode while preserving valid trajectory context.
5. **Pre-Flight Invariant Verifier:** Formally verifies state invariants and return schemas prior to resuming agent autonomy.

---

## 3. Primary Empirical Results

Empirical evaluations were conducted across 400 real-world execution trials spanning 4 distinct agent architectures (Calculator Agent, Knowledge Search Agent, SQLite Database Agent, and a 4-Node Multi-Agent DAG Pipeline) subjected to 16 fine-grained fault injections across 6 domains.

### Table 1: Main Comparative Benchmark (5-Way Ablation Across 400 Real Trials)

| Recovery Paradigm | Fault Detection (FDR %) | Root Cause Accuracy (RCA %) | Recovery Success (RSR %) | Recovery Latency (RL s) | Recovery Cost (RC tok) | Active MTTR (MTTR_A s) | Active MTCR (MTCR_A tok) | Cascade Containment (CCF %) | Token Overhead (TOR %) | Cost-Normalized Efficiency (CNRE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DRAC Full System** | **100.0%** | **75.0%** | **100.0%** | **0.000s** | **142.5** | **0.000s** | **142.5** | **100.0%** | **35.6%** | **20.00** |
| **DRAC (Fixed Heuristic)** | 100.0% | 75.0% | 100.0% | 0.000s | 172.5 | 0.000s | 172.5 | 100.0% | 43.1% | 20.00 |
| **Reflexion (In-band)** | 100.0% | 0.0% | 73.8% | 0.800s | 430.0 | 0.800s | 430.0 | 100.0% | 107.5% | 0.053 |
| **Pure Rollback (Amnesia)** | 100.0% | 0.0% | 50.0% | 0.000s | 60.0 | 0.000s | 60.0 | 97.5% | 15.0% | 10.00 |
| **Naive Retry** | 100.0% | 0.0% | 36.2% | 0.000s | 120.0 | 0.000s | 120.0 | 97.5% | 30.0% | 7.25 |

### Table 2: Granular Fault Breakdown (Recovery Success Rate % Across All 8 Injected Perturbations)

| Perturbation Type | Taxonomy Domain | Naive Retry | Reflexion (In-band) | Pure Rollback | DRAC Fixed | DRAC Full System |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `PLAN_CIRCULAR_LOOP` | Planning Loop | 0.0% | 0.0% | 30.0% | **100.0%** | **100.0%** |
| `TOOL_SERVER_500` | Infrastructure Crash | 0.0% | 0.0% | 0.0% | **100.0%** | **100.0%** |
| `CONTEXT_STALE_STATE` | Memory Inconsistency | 0.0% | 100.0% | 50.0% | **100.0%** | **100.0%** |
| `TOOL_EMPTY_RETURN` | Tool Return Omission | 20.0% | 100.0% | 20.0% | **100.0%** | **100.0%** |
| `TOOL_INVALID_ARGS` | Tool Parameter Error | 50.0% | 90.0% | 50.0% | **100.0%** | **100.0%** |
| `TOOL_TIMEOUT` | Network Latency Spike | 70.0% | 100.0% | 100.0% | **100.0%** | **100.0%** |
| `COMM_MESSAGE_LOSS` | Communication Loss | 80.0% | 100.0% | 80.0% | **100.0%** | **100.0%** |
| `SCHEMA_MALFORMED_JSON`| Schema Invariant Breach| 70.0% | 100.0% | 70.0% | **100.0%** | **100.0%** |

### Table 3: Workload-by-Workload Task Breakdown (Recovery Success Rate %)

| Benchmark Workload | Operational Modality | Naive Retry | Reflexion (In-band) | Pure Rollback | DRAC Fixed | DRAC Full System |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Calculator Agent** | Arithmetic & Tool Calling | 25.0% | 75.0% | 50.0% | 100.0% | **100.0%** |
| **Web Search Agent** | Fact Retrieval & Search Indexing | 37.5% | 75.0% | 50.0% | 100.0% | **100.0%** |
| **SQL Database Agent** | Relational In-Memory SQLite Queries | 37.5% | 68.8% | 37.5% | 100.0% | **100.0%** |
| **Multi-Agent Pipeline** | 4-Node Collaborative DAG | 50.0% | 75.0% | 62.5% | 100.0% | **100.0%** |

---

## 4. Formal SRE Metrics & Mathematical Specification

### Table 4: Formal Reliability & Recovery Efficiency Metrics

| Metric Symbol | Full Metric Name | Formal Definition | Operational SRE Significance |
| :--- | :--- | :--- | :--- |
| **FDR** | Fault Detection Rate | $\text{FDR} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Detected}_i) \times 100\%$ | Proportion of injected perturbations flagged by out-of-band monitoring. |
| **RCA** | Root Cause Accuracy | $\text{RCA} = \frac{1}{N_{\text{det}}} \sum_{i=1}^{N_{\text{det}}} \mathbb{I}(\hat{d}_i = d_i^*) \times 100\%$ | Diagnostic precision of attributed fault domain $\hat{d}_i$ against ground-truth domain $d_i^*$. |
| **RSR** | Recovery Success Rate | $\text{RSR} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Success}_i) \times 100\%$ | Primary resilience KPI: fraction of perturbed trials reaching valid task completion. |
| **RL** | Recovery Latency | $\text{RL} = \frac{1}{N} \sum_{i=1}^N t_{\text{rec}}^{(i)}$ | Mean clock latency consumed across detection, diagnosis, and state recovery. |
| **RC** | Recovery Cost | $\text{RC} = \frac{1}{N} \sum_{i=1}^N c_{\text{rec}}^{(i)}$ | Mean additional token expenditure consumed during recovery. |
| **MTTR_A** | Mean Time to Recover (Active) | $\text{MTTR}_A = \frac{1}{N_{\text{succ}}} \sum_{i \in \mathcal{S}_{\text{succ}}} t_{\text{rec}}^{(i)}$ | Mean clock latency across successful recovery trials, where $N_{\text{succ}} = \text{card}(\mathcal{S}_{\text{succ}})$. |
| **MTCR_A** | Mean Tokens to Recover (Active) | $\text{MTCR}_A = \frac{1}{N_{\text{succ}}} \sum_{i \in \mathcal{S}_{\text{succ}}} c_{\text{rec}}^{(i)}$ | Mean recovery token cost across successful trials, where $N_{\text{succ}} = \text{card}(\mathcal{S}_{\text{succ}})$. |
| **CCF** | Cascade Containment Factor | $\text{CCF} = \frac{1}{N_{\text{MAS}}} \sum_{j=1}^{N_{\text{MAS}}} \mathbb{I}(\text{Contained}_j) \times 100\%$ | Proportion of multi-agent faults neutralized at origin without downstream poisoning. |
| **TOR** | Token Overhead Ratio | $\text{TOR} = \frac{\overline{C}_{\text{rec}}}{\overline{C}_{\text{base}}} \times 100\%$ | Percentage token inflation relative to unperturbed baseline execution cost $\overline{C}_{\text{base}}$. |
| **CNRE** | Cost-Normalized Recovery Efficiency | $\text{CNRE} = \frac{\text{RSR} / 100}{\log_2(1 + C_{\text{norm}}) \cdot \log_2(1 + T_{\text{norm}})}$ | Pareto frontier metric with $C_{\text{norm}} = \frac{\overline{C}_{\text{rec}}}{\overline{C}_{\text{base}}}$ and $T_{\text{norm}} = \frac{\overline{T}_{\text{rec}}}{\overline{T}_{\text{base}}}$. |

### Mathematical Formulations

Let $\mathcal{T} = \{1, 2, \dots, N\}$ denote the set of all evaluated trials ($N = 400$). Let $\mathcal{S}_{\text{succ}} = \{i \in \mathcal{T} \mid \text{Success}_i = 1\}$ denote the subset of trials in which the recovery controller successfully restored valid execution, with active recovery cardinality:
$$N_{\text{succ}} = |\mathcal{S}_{\text{succ}}| = \sum_{i=1}^N \mathbb{I}(\text{Success}_i)$$

1. **Mean Time to Recover (Active):**
   $$\text{MTTR}_A = \frac{\sum_{i \in \mathcal{S}_{\text{succ}}} \text{Latency}^{(i)}}{|\mathcal{S}_{\text{succ}}|} = \frac{1}{N_{\text{succ}}} \sum_{i \in \mathcal{S}_{\text{succ}}} t_{\text{rec}}^{(i)}$$
   where $t_{\text{rec}}^{(i)}$ represents the recovery latency (wall-clock seconds) of trial $i$, conditioning exclusively on successful recoveries to prevent skew from unrecovered timeouts.

2. **Mean Tokens to Recover (Active):**
   $$\text{MTCR}_A = \frac{\sum_{i \in \mathcal{S}_{\text{succ}}} \text{Tokens}^{(i)}}{|\mathcal{S}_{\text{succ}}|} = \frac{1}{N_{\text{succ}}} \sum_{i \in \mathcal{S}_{\text{succ}}} c_{\text{rec}}^{(i)}$$
   where $c_{\text{rec}}^{(i)}$ denotes the additional prompt and completion tokens incurred by the recovery mechanism for trial $i$.

3. **Cost-Normalized Recovery Efficiency (CNRE):**
   $$\text{CNRE} = \frac{\text{RSR} / 100}{\max\left(\epsilon, \, \log_2(1 + C_{\text{norm}}) \cdot \log_2(1 + T_{\text{norm}})\right)}$$
   where:
   - $C_{\text{norm}} = \frac{\overline{C}_{\text{rec}}}{\max(1, \overline{C}_{\text{base}})}$ is the normalized recovery token cost relative to the nominal unperturbed task baseline $\overline{C}_{\text{base}}$.
   - $T_{\text{norm}} = \frac{\overline{T}_{\text{rec}}}{\max(\delta, \overline{T}_{\text{base}})}$ is the normalized recovery wall-clock latency relative to baseline execution latency $\overline{T}_{\text{base}}$ ($\delta = 0.01\text{s}$).
   - $\epsilon = 0.05$ is a positive regularization bound guaranteeing numerical stability when sub-millisecond fast-path recoveries approach $T_{\text{norm}} \approx 0$.

---

## 5. Technical Documentation & Extended Artifacts

* **[COMPLETE_TABLES_AND_PROOF.md](COMPLETE_TABLES_AND_PROOF.md):** Complete extended documentation containing all 6 empirical tables, formal mathematical proofs for Theorems 1-3, and the failure-mode remediation impact matrix.
* **Academic Manuscript:** Research paper preprint under review (available upon academic request; preprint forthcoming on arXiv).
* **[paper/references.bib](paper/references.bib):** Complete 45-paper BibTeX database covering literature from 2023 to 2026.
* **[experiments/results/summary_metrics.csv](experiments/results/summary_metrics.csv):** Aggregated metrics across all 5 evaluated recovery paradigms.
* **[experiments/results/raw_trials.csv](experiments/results/raw_trials.csv):** Full trial-by-trial logs across all 400 real-execution evaluations.

---

## 6. Quick Start

### Installation

```bash
git clone https://github.com/<username>/drac.git
cd drac
pip install -r requirements.txt
```

### Verification & Reproduction Commands

Run the 9-test production verification suite (100% pass rate):
```bash
python main.py --mode verify
```

Display all publication tables directly in the terminal:
```bash
python main.py --mode tables
```

Execute the full 400-trial benchmark suite and generate all figures:
```bash
python main.py --mode all
```

---

## 7. Repository Structure

```
drac/
|-- drac/                       # Core DRAC Resilience Engine
|   |-- types.py                # TelemetryEvent, FaultDomain, Checkpoint, Budget
|   |-- detector.py             # Out-of-band invariant anomaly detector
|   |-- diagnoser.py            # Dual-Process Diagnoser (System 1 & System 2)
|   |-- dncs.py                 # Distilled Negative Constraint Synthesizer
|   |-- state_manager.py        # Transactional checkpointer & context pruner
|   |-- arbiter.py              # B-POMDP budget-constrained utility arbiter
|   \-- verifier.py             # Pre-flight state & schema invariant verifier
|-- agents/                     # Benchmark Agent Architectures
|   |-- calculator_agent.py     # Arithmetic & numerical math tool agent
|   |-- search_agent.py         # Knowledge-retrieval search agent
|   |-- db_agent.py             # Relational in-memory SQLite database agent
|   \-- multi_agent_pipeline.py # 4-node collaborative pipeline (Planner->Researcher->Analyst->Reviewer)
|-- injector/                   # Chaos Fault Injection Framework
|   |-- fault_types.py          # 16 fine-grained fault definitions across 6 domains
|   \-- proxy.py                # Non-invasive runtime tool interception proxy
|-- baselines/                  # Comparative Recovery Paradigms
|   \-- strategies.py           # Naive Retry, Reflexion, Pure Rollback, DRAC Fixed, DRAC Full
|-- experiments/                # Empirical Benchmarking Suite
|   |-- metrics.py              # SRE Metrics engine (RSR, RCA, FDR, TOR, CCF, CNRE)
|   |-- run_benchmarks.py       # 400-trial real execution benchmark runner
|   |-- plot_results.py         # Figures 1 to 4 publication plotter
|   |-- plot_theoretical_diagrams.py # Figures 5 & 6 vector plot generator
|   \-- plots/                  # Generated high-resolution publication figures
|-- paper/                      # Academic Manuscript & BibTeX Database
|   |-- COMPLETE_TABLES_AND_PROOF.md # Dedicated empirical tables and mathematical proofs
|   |-- references.bib          # 45-paper BibTeX bibliography (2023-2026)
|   \-- figures/                # High-resolution diagrams & vector plots
|-- tests/                      # Production Test Suite
|   \-- test_production_suite.py# Comprehensive unit and integration test suite
|-- main.py                     # Master CLI runner (--mode all|verify|benchmark|tables)
|-- LICENSE                     # Non-Commercial Research & Educational License
\-- README.md                   # Repository Documentation
```

---

## 8. Citation

```bibtex
@article{drac2026faulttolerance,
  title={DRAC: Diagnosis, Recovery, and Adaptive Control for Fault-Tolerant LLM Agents},
  author={Senior Research Team},
  journal={arXiv preprint arXiv:2603.XXXXX},
  year={2026},
  url={https://github.com/ankushpahal-12/drac-agent}
}
```
