# AgentAlign Lab

A research-grade pipeline for verifier-guided preference learning in terminal agents. AgentAlign Lab generates synthetic tasks, runs ReAct-style agent loops, scores trajectories with deterministic verifiers, constructs DPO preference pairs, and fine-tunes a language model via QLoRA — producing an agent that improves through structured feedback rather than human annotation at scale.

---

## Motivation

Current RLHF pipelines depend heavily on human preference labels, which are expensive, inconsistent, and difficult to scale for agentic settings where the action space is broad and task outcomes are verifiable. AgentAlign Lab replaces human raters with deterministic verifiers wherever the task has an unambiguous correct answer — code execution, arithmetic, structured data retrieval, and similar domains — closing the feedback loop automatically and cheaply.

---

## Pipeline Overview

```
Task Generation
      |
      v
ReAct Agent Loop  <---  Tool Calls (code exec, search, etc.)
      |
      v
Deterministic Verifier  -->  Pass / Fail + Score
      |
      v
DPO Preference Pair Construction
      |
      v
QLoRA Fine-Tuning (TRL)
      |
      v
Evaluation + Gradio Dashboard
```

Each stage is modular. You can run the verifier in isolation, swap in a different base model, or inject real task datasets at the generation stage without touching the rest of the pipeline.

---

## Repository Structure

```
AgentAlign-Lab/
├── tasks/                  # Task generators and dataset schemas
├── agent/                  # ReAct loop implementation
├── verifiers/              # Deterministic verifier suite
├── preference/             # DPO pair construction logic
├── training/               # QLoRA fine-tuning via TRL
├── eval/                   # Evaluation harness
├── dashboard/              # Gradio interface
├── configs/                # Experiment configs (YAML)
├── scripts/                # End-to-end run scripts
└── notebooks/              # Exploratory analysis
```

---

## Setup

Requires Python 3.13. Dependencies are managed via `uv`.

```bash
git clone https://github.com/hey-shiv/AgentAlign-Lab.git
cd AgentAlign-Lab
uv sync
```

To run the full pipeline end-to-end:

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

To launch the evaluation dashboard:

```bash
python dashboard/app.py
```

---

## Key Design Decisions

**Deterministic verifiers over learned reward models.** For tasks with ground-truth outputs, a rule-based verifier is more reliable, interpretable, and cheaper than training a separate reward model. Verifiers here are written as pure functions: input is a trajectory, output is a scalar score.

**DPO over PPO.** Direct Preference Optimization avoids the instability and hyperparameter sensitivity of online RL. Given the synthetic pair construction setup, DPO is a natural fit.

**QLoRA for compute-constrained fine-tuning.** The fine-tuning stage targets a small decoder-only base model and runs on Apple Silicon (M-series) or free-tier GPU (Colab/Kaggle). No dedicated GPU required for experimentation.

**Modular stages.** Each component writes its outputs to disk in a defined schema. Stages communicate through files rather than in-memory state, making it easy to inspect intermediate results and resume interrupted runs.

---

## Current Status

Active development. The task generation, ReAct loop, and verifier stages are the primary focus. Fine-tuning and evaluation are scaffolded but not yet fully integrated.

---

## Research Direction

The longer-term research question driving this project: to what extent can verifier-guided feedback, without any human annotation, produce agents that generalize across task types rather than overfitting to the verifier's specific scoring function? AgentAlign Lab is the empirical testbed for that question, targeting a workshop paper or technical report as a first output.

---

## Author

Shiv — B.Tech CS (AI/ML), VIT-AP University  
GitHub: [hey-shiv](https://github.com/hey-shiv)  
X: [@NaadhLabs](https://x.com/NaadhLabs)  
Portfolio: [hey-shiv.github.io](https://hey-shiv.github.io)
