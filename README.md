<div align="center">
  <h1>AgentAlign Lab</h1>
  <p><strong>A verifier-guided preference learning pipeline for terminal agents.</strong></p>

  <p>
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white" alt="Python Version"></a>
    <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/uv-Package_Manager-purple.svg?logo=python" alt="uv Package Manager"></a>
    <a href="https://huggingface.co/docs/trl/index"><img src="https://img.shields.io/badge/TRL-DPOTrainer-yellow.svg?logo=huggingface&logoColor=white" alt="TRL"></a>
    <a href="https://gradio.app/"><img src="https://img.shields.io/badge/Gradio-Dashboard-orange.svg?logo=gradio&logoColor=white" alt="Gradio"></a>
  </p>
</div>

---

**AgentAlign Lab** is a complete, end-to-end pipeline designed to fine-tune small Language Models (like `Qwen2.5-Coder-1.5B-Instruct`) to act as autonomous, reliable ReAct-style terminal agents. It uses deterministic verifiers to automatically score agent trajectories and generate Direct Preference Optimization (DPO) datasets without requiring human labeling.

## Key Features

1. **Task Generation**: Deterministically generates isolated coding and data tasks across 5 distinct families (Python bugfixes, data transformations, config repairs, log extractions, and safety traps).
2. **Agent Execution Loop**: Runs a ReAct-style agent with an allowlisted, sandboxed terminal environment.
3. **Trajectory Logging**: Captures structured execution histories detailing the agent's thoughts, actions, tools used, and environment observations.
4. **Deterministic Verifiers**: Automatically grades the agent's work using `pytest`, JSON schema validation, exact matching, and safety checks.
5. **Preference Pair Creation**: Converts raw scored trajectories into chosen/rejected DPO preference pairs based on strict score margins.
6. **DPO + QLoRA Training**: Fine-tunes the base model using Hugging Face's `TRL` library, optimized with 4-bit quantization (QLoRA) for consumer hardware.
7. **Evaluation Framework**: Robust baseline vs. tuned model comparison to track improvements in success rate, step efficiency, and safety compliance.
8. **Observability Dashboard**: A rich Gradio interface to explore tasks, trace trajectories, analyze failure modes, and visualize metrics.

---

## Tech Stack

- **Core & Schemas**: Python 3.11+, Pydantic v2
- **Environment Management**: `uv`
- **Model Inference**: Hugging Face Transformers
- **Training**: TRL (DPOTrainer) + PEFT (QLoRA)
- **Data Handling**: Hugging Face Datasets, JSONL
- **Verification**: `pytest`, sandboxed `subprocess`
- **UI/Dashboard**: Gradio Blocks

---

## Quickstart

### 1. Installation

Clone the repository and use [`uv`](https://github.com/astral-sh/uv) to sync the dependencies:

```bash
git clone https://github.com/hey-shiv/AgentAlign-Lab.git
cd AgentAlign-Lab

# Create virtual environment and install dependencies
uv sync
```

### 2. Execution Pipeline

The repository includes a numbered sequence of scripts representing the entire lifecycle of the pipeline.

```bash
# 1. Generate the task suite (train/val/test splits)
uv run python scripts/01_generate_tasks.py

# 2. Run the agent to collect raw trajectories
uv run python scripts/02_run_agent.py --split train --agent baseline --model dummy --out runs/train --repetitions 8

# 3. Score the raw trajectories using deterministic verifiers
uv run python scripts/03_score_trajectories.py --runs-dir runs/train --out-dir data/trajectories/scored_train

# 4. Build DPO preference pairs from scored trajectories
uv run python scripts/04_build_preferences.py --runs-dir data/trajectories/scored_train --out data/preferences/dpo_train.jsonl

# 5. Train the DPO adapter (Requires GPU/Colab/Kaggle)
uv run python scripts/05_train_dpo.py

# 6. Evaluate and compare baseline vs. tuned models
uv run python scripts/06_eval_models.py --runs-dir data/trajectories/scored_train --agent baseline

# 7. Launch the Observability Dashboard
uv run python scripts/07_launch_dashboard.py
```

---

## Project Structure

```text
AgentAlign-Lab/
├── configs/                  # YAML configurations for agent, training, and evaluation
├── data/
│   ├── tasks/                # Generated task definitions (JSONL)
│   ├── trajectories/         # Raw and scored agent execution traces
│   └── preferences/          # DPO chosen/rejected pairs
├── scripts/                  # End-to-end execution scripts (01-07)
├── src/
│   └── agentalign/
│       ├── agent/            # ReAct loop, prompts, parser, tools, sandbox
│       ├── dashboard/        # Gradio observability UI
│       ├── data/             # Serialization and preference pair construction
│       ├── eval/             # Metrics, model comparison, failure analysis
│       ├── tasks/            # Task generation and template loading
│       ├── train/            # SFT and DPO training loops with QLoRA
│       └── verifier/         # Deterministic checks, scoring, and safety scans
├── tests/                    # Comprehensive pytest suite covering schemas, parsers, and verifiers
└── pyproject.toml            # Project dependencies and metadata
```

---

## Testing

The pipeline logic is fully tested. You can run the test suite using:

```bash
uv run pytest tests/ -v
```

---

*Designed and implemented as an advanced sandbox for verifiable AI safety and agentic fine-tuning.*
