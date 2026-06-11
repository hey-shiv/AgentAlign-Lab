# AgentAlign Lab

Verifier-guided preference data for reliable terminal agents. This project builds a compact local feedback loop:

tasks -> agent runs -> trajectory logs -> deterministic verifiers -> DPO preference pairs -> evaluation -> dashboard.

## Current Build

- 60 deterministic tasks across 5 families:
  - 25 Python bug fixes
  - 15 data transformation tasks
  - 10 JSON/config repair tasks
  - 5 log extraction tasks
  - 5 safety traps
- 757 scored trajectories currently available across clean train/validation/test artifacts.
- 336 train preference pairs and 9 validation preference pairs.
- Family-specific verifiers: `pytest`, exact JSON, exact file, JSON schema, and safety-aware pytest.
- Split-aware evaluation with pass rate, score, invalid action rate, unsafe action rate, timeout rate, and average steps.
- Gradio dashboard with Overview, Task Explorer, Trajectory Viewer, Preference Pair Viewer, Baseline vs Tuned, and Failure Analysis tabs.
- DPO training script with local dataset dry-run validation and a TRL/QLoRA training path for GPU machines.

## Setup

```bash
uv pip install -e ".[dev,dashboard]"
```

For actual QLoRA/DPO training, also install the ML stack on a GPU machine:

```bash
uv pip install -e ".[ml]"
```

## Reproduce The Local Pipeline

Generate the task suite:

```bash
python scripts/01_generate_tasks.py
```

Collect train trajectories with the scripted baseline and failure agent:

```bash
python scripts/02_run_agent.py --split train --agent baseline --model dummy --out runs/train --repetitions 8 --clear
python scripts/02_run_agent.py --split train --agent bad_model --model bad_dummy --out runs/train --repetitions 8 --clear
```

Score trajectories and build DPO pairs:

```bash
python scripts/03_score_trajectories.py --runs-dir runs/train --out-dir data/trajectories/scored_train --clear
python scripts/04_build_preferences.py --runs-dir data/trajectories/scored_train --out data/preferences/dpo_train.jsonl --split train --min-margin 2.0
```

Validate the DPO dataset locally:

```bash
python scripts/05_train_dpo.py --dry-run
python scripts/05_train_dpo.py --preflight
```

Evaluate:

```bash
python scripts/06_eval_models.py --runs-dir data/trajectories/scored_train --split train --agent baseline --compare-agent bad_model
python scripts/06_eval_models.py --runs-dir data/trajectories/scored_test --split test --agent baseline --compare-agent bad_model
```

Audit the local MVP artifacts:

```bash
python scripts/08_audit_mvp.py
```

Prepare a portable CUDA handoff bundle:

```bash
python scripts/09_prepare_gpu_handoff.py --clear
```

This writes `outputs/gpu_handoff/` and `outputs/gpu_handoff.zip` with the runnable mini-repo, DPO data, scored trajectory evidence, config, scripts, manifest, Colab notebook, and GPU runbook. In Colab, upload the zip and open `notebooks/02_colab_dpo_training.ipynb`. The audit works both in the local source tree and inside the extracted Colab bundle.

Launch the dashboard:

```bash
python scripts/07_launch_dashboard.py
```

The launcher prints the local URL and chooses a free port automatically.

## Architecture

- **Task Runner:** Creates isolated `tempdir` workspaces from task JSON files.
- **Agent Loop:** Single-agent loop with strict JSON actions and max 8 steps.
- **Tools:** `list_files`, `read_file`, `write_file`, `run_command`, and `final_answer`.
- **Sandbox:** Allowlisted subprocess commands, no shell execution, timeouts, and workspace path checks.
- **Verifier:** Deterministic correctness, schema/output, and safety checks.
- **Preference Builder:** High-margin chosen/rejected trajectory pairs in DPO JSONL format.
- **Evaluation:** Split-aware baseline/comparison metrics.
- **Dashboard:** Local Gradio observability UI.

## Important Limitation

The repository currently validates the DPO dataset and training entrypoint locally. A real LoRA adapter requires running `scripts/05_train_dpo.py` without `--dry-run` on a CUDA GPU machine. The local preflight reports whether the current environment is ready.

The expected adapter files after GPU training are:

- `outputs/adapters/qwen_dpo_final/adapter_config.json`
- `outputs/adapters/qwen_dpo_final/adapter_model.safetensors`

The GPU training defaults come from `configs/dpo_qwen15_lora.yaml`; the checked-in config targets `Qwen/Qwen2.5-Coder-1.5B-Instruct` with LoRA output at `outputs/adapters/qwen_dpo_final`.

After the adapter is trained, run held-out base-vs-adapter evaluation in the same GPU environment:

```bash
python scripts/02_run_agent.py --split test --agent qwen_base --model hf --out runs/test_qwen --clear
python scripts/02_run_agent.py --split test --agent qwen_dpo --model hf_adapter --adapter-path outputs/adapters/qwen_dpo_final --out runs/test_qwen --clear
python scripts/03_score_trajectories.py --runs-dir runs/test_qwen --out-dir data/trajectories/scored_test_qwen --clear
python scripts/06_eval_models.py --runs-dir data/trajectories/scored_test_qwen --split test --agent qwen_dpo --compare-agent qwen_base
python scripts/08_audit_mvp.py
```

## License

MIT
