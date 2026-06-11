# AgentAlign Lab: Beginner-To-Interview Technical Guide

This document is a detailed study and execution guide for AgentAlign Lab.

It is written for a beginner who wants to understand the project deeply enough to:

- run it locally,
- run the GPU training/evaluation part in Google Colab,
- debug common failures,
- explain every major design choice in a technical interview,
- answer questions about agents, verifiers, DPO, LoRA/QLoRA, evaluation, and limitations.

You do not need to understand everything in one sitting. Read it in passes.

Recommended reading passes:

1. First pass: read Sections 1-5 to understand the project story.
2. Second pass: run Sections 8-11 locally.
3. Third pass: run Sections 12-14 in Colab.
4. Fourth pass: study Sections 15-22 for interview explanations.

## 1. One-Sentence Project Explanation

AgentAlign Lab is a compact verifier-guided post-training pipeline for terminal agents: it runs agents on small terminal tasks, logs their tool-use trajectories, scores them with deterministic verifiers, converts better/worse attempts into DPO preference data, trains a LoRA adapter on Colab, and evaluates whether the tuned model improves on held-out tasks.

If an interviewer asks, "What did you build?", say:

> I built a small end-to-end AI systems project for improving terminal-agent reliability. It includes a task suite, sandboxed agent loop, trajectory logging, deterministic verifiers, preference-pair generation, DPO/QLoRA training, held-out evaluation, and a Gradio dashboard for trace and failure analysis.

## 2. The Core Problem

LLM agents often fail in ways that are not visible from the final answer alone.

They may:

- call invalid tools,
- write malformed JSON,
- edit the wrong file,
- pass the wrong command,
- delete protected tests,
- loop until timeout,
- produce a confident final answer while the task is still wrong.

The project asks:

```text
Can deterministic verifier feedback create useful preference data for improving terminal-agent reliability?
```

This matters because modern AI companies need loops like:

```text
run agent -> trace behavior -> score outcome -> curate data -> improve model/system -> evaluate regression
```

AgentAlign Lab builds this loop at a small, reproducible scale.

## 3. What The Project Is Not

This project is intentionally not:

- a production coding agent,
- a full SWE-bench clone,
- a huge RLHF system,
- a browser automation benchmark,
- a multi-agent framework,
- a production sandbox,
- a LangSmith/Phoenix clone.

It is a compact proof-of-work for:

- agent evaluation,
- verifier engineering,
- trace observability,
- preference-data generation,
- small-model post-training,
- held-out reliability measurement.

In an interview, this is important. Do not oversell it as a universal benchmark. Say:

> I intentionally kept the task distribution small and deterministic so the full feedback loop is reproducible and auditable.

## 4. Project Flow

The main workflow is:

```text
1. Generate task suite
2. Run agents on tasks
3. Save trajectories
4. Score trajectories with verifiers
5. Build DPO preference pairs
6. Validate DPO data locally
7. Train LoRA adapter on Colab GPU
8. Run held-out base-vs-adapter evaluation
9. Inspect dashboard and report
```

The artifact flow is:

```text
data/tasks/*.json
    -> runs/**/*.jsonl
    -> data/trajectories/scored_*/**/*.jsonl
    -> data/preferences/dpo_train.jsonl
    -> outputs/adapters/qwen_dpo_final
    -> data/trajectories/scored_test_qwen
```

The strongest mental model:

```text
claim -> command -> artifact -> metric -> explanation
```

For every claim in your README/report/interview, you should know:

- which command produced it,
- which file proves it,
- which metric summarizes it,
- what limitation applies.

## 5. Current Build Status

Current local artifacts:

- 60 tasks across 5 families.
- 757 scored trajectories.
- 336 train preference pairs.
- 9 validation preference pairs.
- Train/validation/test split by task ID.
- Local DPO dry-run validation.
- Colab handoff bundle generator.
- Colab notebook for GPU training.
- Audit script that verifies local and Colab handoff readiness.
- Dashboard with task, trajectory, preference, metric, and failure views.

Expected remaining full-MVP artifacts after Colab:

```text
outputs/adapters/qwen_dpo_final/adapter_config.json
outputs/adapters/qwen_dpo_final/adapter_model.safetensors
data/trajectories/scored_test_qwen/*.jsonl
runs/test_qwen/*.jsonl
```

The local machine may not have CUDA. That is okay. Colab handles the GPU part.

## 6. What To Study

Use this reading plan. It is intentionally practical. Do not read everything end-to-end before running the project.

### Core Project Resources

| Resource | Link/File | Read Amount | Why |
|---|---|---:|---|
| Project README | `README.md` | 15 minutes | Fast overview, commands, current status |
| Technical report | `report/paper.md` | 30-45 minutes | Interview narrative, metrics, limitations |
| Handbook PDF | `guide/agentalign_lab_master_execution_handbook_v3.pdf` | 60-90 minutes | Original plan and success criteria |
| Colab notebook | `notebooks/02_colab_dpo_training.ipynb` | 20 minutes before Colab | GPU execution flow |
| Audit script | `scripts/08_audit_mvp.py` | 20 minutes | What "ready" means in code |

Important handbook sections:

- Executive Summary: 10 minutes
- MVP Success Criteria: 10 minutes
- Evaluation Integrity: 15 minutes
- Data Flywheel: 10 minutes
- Week 1-6 Roadmap: 20 minutes
- Quality Control Checklists: 20 minutes
- Skip optional V2/V3 ideas until the MVP is complete.

### Agent And Tool-Use Resources

| Resource | Link/File | Read Amount | Focus |
|---|---|---:|---|
| Local agent loop | `src/agentalign/agent/loop.py` | 30 minutes | How prompt, model, parser, tools, verifier connect |
| Parser | `src/agentalign/agent/parser.py` | 20 minutes | Strict JSON action parsing |
| Tools | `src/agentalign/agent/tools.py` | 30 minutes | Command allowlist, workspace safety |
| Prompt | `src/agentalign/agent/prompts.py` | 15 minutes | How model sees the task and tool schema |
| Python subprocess docs | https://docs.python.org/3/library/subprocess.html | 20 minutes | Why `shell=False` and timeouts matter |

What to understand:

- why strict JSON reduces ambiguity,
- why invalid action rate is a metric,
- why file access must stay inside the workspace,
- why commands are allowlisted,
- why every step is logged.

### Schema And Data Resources

| Resource | Link/File | Read Amount | Focus |
|---|---|---:|---|
| Schemas | `src/agentalign/schemas.py` | 45 minutes | Task, Step, Trajectory, VerifierResult, PreferencePair |
| Pydantic models | https://docs.pydantic.dev/latest/concepts/models/ | 30 minutes | `BaseModel`, validation, `model_validate_json` |
| JSON Lines | https://jsonlines.org/ | 10 minutes | Why `.jsonl` is useful for trajectories/preferences |
| jsonlines Python package | https://jsonlines.readthedocs.io/en/latest/ | 10 minutes | Reading/writing line-delimited JSON |

Interview explanation:

> I used Pydantic schemas so artifacts are validated at boundaries. Tasks, trajectories, verifier results, and preference pairs all have structured contracts, which makes the pipeline easier to test and debug.

### Verifier Resources

| Resource | Link/File | Read Amount | Focus |
|---|---|---:|---|
| Verifier checks | `src/agentalign/verifier/checks.py` | 45 minutes | Family-specific deterministic checks |
| Scoring | `src/agentalign/verifier/score.py` | 30 minutes | Composite score and behavior penalties |
| Family verifier tests | `tests/test_family_verifiers.py` | 20 minutes | False positive/negative examples |
| pytest docs | https://docs.pytest.org/en/stable/ | 30 minutes | How Python tasks are verified |

What to understand:

- deterministic verifiers are the feedback source,
- pass/fail is not enough, so scores include behavior penalties,
- safety tasks check forbidden behavior,
- verifiers can be gamed, so held-out tasks and anti-cheat checks matter.

### Preference Learning Resources

| Resource | Link/File | Read Amount | Focus |
|---|---|---:|---|
| Preference builder | `scripts/04_build_preferences.py` | 30 minutes | How chosen/rejected pairs are created |
| Preference data code | `src/agentalign/data/preferences.py` | 20 minutes | Data formatting |
| DPO paper | https://arxiv.org/abs/2305.18290 | 45-60 minutes | Abstract, intro, method overview |
| Hugging Face TRL docs | https://huggingface.co/docs/trl | 20 minutes | Library purpose |
| TRL DPOTrainer docs | https://huggingface.co/docs/trl/main/dpo_trainer | 30-45 minutes | Expected data format and trainer config |

How much of the DPO paper to read:

- Read abstract: yes.
- Read introduction: yes.
- Read method intuition: yes.
- Skip dense derivations on first pass.
- Read experiments only enough to understand what DPO is compared against.

Beginner explanation of DPO:

> DPO trains a model from preference pairs. For the same prompt, we show a chosen response and a rejected response. The objective increases the probability of the chosen response relative to the rejected one. In this project, the preference pairs come from verifier scores instead of human labels.

### LoRA, QLoRA, And GPU Training Resources

| Resource | Link/File | Read Amount | Focus |
|---|---|---:|---|
| Training script | `scripts/05_train_dpo.py` | 60 minutes | Dataset validation, preflight, model load, LoRA config, DPOTrainer |
| DPO config | `configs/dpo_qwen15_lora.yaml` | 20 minutes | Hyperparameters |
| LoRA paper | https://arxiv.org/abs/2106.09685 | 45 minutes | Abstract, intro, method figure |
| QLoRA paper | https://arxiv.org/abs/2305.14314 | 45 minutes | Abstract, intro, memory idea |
| PEFT docs | https://huggingface.co/docs/peft/index | 30 minutes | How adapters work |
| Transformers PEFT docs | https://huggingface.co/docs/transformers/peft | 20 minutes | Loading adapters in Transformers |
| bitsandbytes quantization docs | https://huggingface.co/docs/transformers/quantization/bitsandbytes | 20 minutes | 4-bit/8-bit quantization basics |
| PyTorch CUDA docs | https://docs.pytorch.org/docs/stable/cuda.html | 20 minutes | CUDA availability and GPU tensors |

How much to read:

- LoRA paper: 45 minutes, focus on why train small adapter matrices.
- QLoRA paper: 45 minutes, focus on 4-bit quantization and memory savings.
- PEFT docs: 30 minutes, focus on `LoraConfig`, adapter save/load.
- Skip advanced LoRA variants until the project is complete.

Beginner explanation:

> Full fine-tuning updates all model weights, which is expensive. LoRA freezes the base model and trains small low-rank adapter matrices. QLoRA makes this cheaper by loading the base model in quantized form while training LoRA adapters. That is why the project can train a small adapter on Colab instead of needing a large GPU machine.

### Hugging Face Inference Resources

| Resource | Link/File | Read Amount | Focus |
|---|---|---:|---|
| Agent runner | `scripts/02_run_agent.py` | 45 minutes | Scripted, base HF, and adapter HF model modes |
| Loading models | https://huggingface.co/docs/transformers/main/models | 20 minutes | `from_pretrained` |
| Pipelines | https://huggingface.co/docs/transformers/main_classes/pipelines | 20 minutes | Text generation pipeline |
| Chat templates | https://huggingface.co/docs/transformers/chat_templating | 30 minutes | Formatting chat prompts correctly |
| Datasets loading | https://huggingface.co/docs/datasets/loading | 20 minutes | Loading local JSON/JSONL |

What to understand:

- base model and adapter model must use the same task split,
- same tools and max steps must be used for fair eval,
- chat templates matter because chat models expect a specific prompt format.

### Dashboard And Reporting Resources

| Resource | Link/File | Read Amount | Focus |
|---|---|---:|---|
| Dashboard app | `src/agentalign/dashboard/app.py` | 45 minutes | How metrics and trace views are loaded |
| Dashboard launcher | `scripts/07_launch_dashboard.py` | 10 minutes | Local port selection |
| Gradio Blocks docs | https://www.gradio.app/docs/gradio/blocks | 20 minutes | Tabs and components |
| Technical report | `report/paper.md` | 45 minutes | How to present results honestly |

What to understand:

- dashboard should show real artifacts, not fake metrics,
- failure analysis is more valuable than decorative UI,
- results must include denominators and limitations.

### Code Quality Resources

| Resource | Link/File | Read Amount | Focus |
|---|---|---:|---|
| Ruff linter docs | https://docs.astral.sh/ruff/linter/ | 15 minutes | `ruff check .` |
| Ruff formatter docs | https://docs.astral.sh/ruff/formatter/ | 10 minutes | Formatting if needed |
| pytest docs | https://docs.pytest.org/en/stable/ | 30 minutes | Running and writing tests |
| Python pathlib docs | https://docs.python.org/3/library/pathlib.html | 15 minutes | Safe file paths |
| Python argparse docs | https://docs.python.org/3/library/argparse.html | 15 minutes | Script CLIs |

Do not spend days here. Read only enough to understand the project commands and tests.

### Beginner Prerequisite Resources

Use these only if you feel shaky on the basics. They are not all required before running the project.

| Topic | Resource | Read Amount | What To Learn |
|---|---|---:|---|
| Python basics | https://docs.python.org/3/tutorial/ | 2-4 hours selectively | functions, imports, files, exceptions |
| Python pathlib | https://docs.python.org/3/library/pathlib.html | 20 minutes | path joins, reading/writing files safely |
| Python argparse | https://docs.python.org/3/library/argparse.html | 20 minutes | how scripts expose CLI flags |
| Python subprocess | https://docs.python.org/3/library/subprocess.html | 30 minutes | command execution, `shell=False`, timeouts |
| JSON | https://www.json.org/json-en.html | 15 minutes | objects, arrays, strings, numbers |
| YAML | https://yaml.org/spec/1.2.2/ | 15 minutes skim | config files and indentation |
| Git book | https://git-scm.com/book/en/v2 | 1-2 hours selectively | status, diff, commits, branches |
| uv docs | https://docs.astral.sh/uv/ | 30 minutes | package/environment management |
| pip docs | https://pip.pypa.io/en/stable/ | 20 minutes | installing packages |
| venv docs | https://docs.python.org/3/library/venv.html | 15 minutes | virtual environments |

Minimum beginner checklist:

- You can run `cd`, `ls`, `python script.py`.
- You know what a virtual environment is.
- You know the difference between `.json`, `.jsonl`, `.yaml`, `.py`, and `.ipynb`.
- You can read a traceback from top to bottom and identify the final error line.

### Machine Learning Background Resources

| Topic | Resource | Read Amount | What To Learn |
|---|---|---:|---|
| Transformers overview | https://huggingface.co/docs/transformers/index | 45 minutes | model/tokenizer/pipeline basics |
| Causal language modeling | https://huggingface.co/docs/transformers/tasks/language_modeling | 30 minutes | next-token prediction intuition |
| Tokenizers | https://huggingface.co/docs/tokenizers/index | 20 minutes | text to tokens and back |
| PyTorch basics | https://pytorch.org/tutorials/beginner/basics/intro.html | 1-2 hours | tensors, modules, training loop |
| CUDA in PyTorch | https://docs.pytorch.org/docs/stable/notes/cuda.html | 30 minutes | CPU vs GPU tensors |

You do not need to become a PyTorch expert for this project. You need to explain:

- what model loading means,
- why GPU memory matters,
- why quantization helps,
- why adapter training is cheaper than full fine-tuning.

### Evaluation And Observability Resources

| Topic | Resource | Read Amount | What To Learn |
|---|---|---:|---|
| OpenAI Evals repo | https://github.com/openai/evals | 30 minutes skim | eval structure and reproducibility mindset |
| HELM paper/site | https://crfm.stanford.edu/helm/latest/ | 30 minutes skim | broad model evaluation framing |
| LangSmith tracing docs | https://docs.smith.langchain.com/ | 20 minutes skim | modern trace observability concepts |
| Arize Phoenix docs | https://docs.arize.com/phoenix | 20 minutes skim | LLM tracing/evals vocabulary |

Read these after you understand the local dashboard. They are useful for interview vocabulary, not required for MVP implementation.

### Related Benchmark Resources

| Benchmark | Resource | Read Amount | What To Learn |
|---|---|---:|---|
| SWE-bench | https://www.swebench.com/ | 20 minutes skim | real-world software engineering evals |
| Terminal-Bench | https://www.tbench.ai/ | 20 minutes skim | terminal-agent task framing |
| HumanEval | https://github.com/openai/human-eval | 20 minutes skim | code-generation benchmark basics |

Use these carefully in interviews. Say:

> My project is not trying to replicate these benchmarks. It is a compact local feedback loop inspired by the same need for reproducible agent evaluation.

### Demo And Interview Prep Resources

| Topic | Resource | Read Amount | What To Learn |
|---|---|---:|---|
| GitHub README guide | https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes | 20 minutes | readable repo presentation |
| Google Colab intro | https://colab.research.google.com/notebooks/intro.ipynb | 20 minutes | notebook execution basics |
| Gradio quickstart | https://www.gradio.app/guides/quickstart | 20 minutes | how local demos work |

For interview prep, spend more time practicing your explanation than reading more links. A good rule:

```text
70% project artifacts, 20% concepts, 10% external resources
```

## 7. Repository Map

Important folders:

```text
configs/                 YAML configs for training/eval
data/tasks/              task JSON files and split JSONLs
data/preferences/        DPO train/val JSONL pairs
data/trajectories/       scored trajectory artifacts
notebooks/               Colab notebook
outputs/                 generated adapters, handoff bundles, metadata
report/                  paper-style project report
scripts/                 runnable pipeline commands
src/agentalign/          main Python package
tests/                   unit tests
guide/                   original handbook PDF
```

Important scripts:

| Script | Purpose |
|---|---|
| `scripts/01_generate_tasks.py` | Generate deterministic task suite |
| `scripts/02_run_agent.py` | Run scripted/HF/HF-adapter agents |
| `scripts/03_score_trajectories.py` | Score raw runs with verifiers |
| `scripts/04_build_preferences.py` | Build chosen/rejected DPO pairs |
| `scripts/05_train_dpo.py` | Validate DPO data and train LoRA adapter |
| `scripts/06_eval_models.py` | Aggregate split-aware metrics |
| `scripts/07_launch_dashboard.py` | Launch Gradio dashboard |
| `scripts/08_audit_mvp.py` | Verify project readiness/artifacts |
| `scripts/09_prepare_gpu_handoff.py` | Build Colab upload bundle |
| `scripts/10_import_colab_artifacts.py` | Validate and import artifacts downloaded from Colab |

## 8. Local Setup

From your terminal:

```bash
cd /Users/shivashant/projects/AgentAlign-Lab
```

Install local dev/dashboard dependencies:

```bash
uv pip install -e ".[dev,dashboard]"
```

Install ML dependencies locally if you want dry-run/preflight support:

```bash
uv pip install -e ".[ml]"
```

Important:

```text
Local ML dependencies != local GPU training.
```

If your machine has no NVIDIA CUDA GPU, real QLoRA training should happen in Colab.

## 9. Local Pipeline Commands

Run these locally from:

```bash
/Users/shivashant/projects/AgentAlign-Lab
```

### 9.1 Generate Tasks

```bash
python scripts/01_generate_tasks.py
```

Expected:

```text
data/tasks/*.json
data/tasks/train.jsonl
data/tasks/val.jsonl
data/tasks/test.jsonl
```

What this proves:

- the task suite exists,
- each task has a schema,
- split files exist.

### 9.2 Run Scripted Train Agents

```bash
python scripts/02_run_agent.py --split train --agent baseline --model dummy --out runs/train --repetitions 8 --clear
python scripts/02_run_agent.py --split train --agent bad_model --model bad_dummy --out runs/train --repetitions 8 --clear
```

What this does:

- `dummy` creates successful reference-like trajectories.
- `bad_dummy` creates failing trajectories.
- multiple repetitions create enough contrast for preference pairs.

Why scripted agents are used locally:

- they validate the pipeline deterministically,
- they produce controlled chosen/rejected examples,
- they do not require GPU or model downloads.

### 9.3 Score Train Trajectories

```bash
python scripts/03_score_trajectories.py --runs-dir runs/train --out-dir data/trajectories/scored_train --clear
```

What this produces:

```text
data/trajectories/scored_train/*.jsonl
```

What this proves:

- verifiers can evaluate trajectories,
- pass/fail and behavior metrics are attached.

### 9.4 Build Train Preference Pairs

```bash
python scripts/04_build_preferences.py --runs-dir data/trajectories/scored_train --out data/preferences/dpo_train.jsonl --split train --min-margin 2.0
```

What this produces:

```text
data/preferences/dpo_train.jsonl
```

Each line contains:

```text
prompt, chosen, rejected, chosen_score, rejected_score, score_margin
```

### 9.5 Validate DPO Data Locally

```bash
python scripts/05_train_dpo.py --dry-run
```

Expected:

```text
outputs/adapters/dpo_run/training_metadata.json
```

This does not train the model. It validates:

- train preference JSONL loads,
- validation preference JSONL loads,
- required fields exist,
- score margins are positive,
- metadata is written.

### 9.6 Check Local Training Readiness

```bash
python scripts/05_train_dpo.py --preflight
```

If local machine has no CUDA, expect something like:

```text
"ready": false
"reason": "QLoRA path requires CUDA; no CUDA accelerator detected"
```

That is not a project failure. It means use Colab.

### 9.7 Evaluate Scripted Baseline

```bash
python scripts/06_eval_models.py --runs-dir data/trajectories/scored_train --split train --agent baseline --compare-agent bad_model
python scripts/06_eval_models.py --runs-dir data/trajectories/scored_test --split test --agent baseline --compare-agent bad_model
```

This proves the evaluation machinery.

Do not claim this is model improvement. It is a pipeline sanity check.

### 9.8 Run Audit

```bash
python scripts/08_audit_mvp.py
```

Before Colab, expected:

- local data/eval loop: PASS
- CUDA training: FAIL locally if no CUDA
- adapter artifacts: FAIL until Colab training
- held-out adapter eval: FAIL until Colab eval

This is honest and expected.

### 9.9 Run Tests And Lint

```bash
ruff check .
pytest
python -m compileall -q src scripts tests
```

These prove:

- code style/import checks pass,
- unit tests pass,
- Python files compile.

## 10. Prepare Colab Bundle Locally

You said the whole folder exists locally. Good. Still, do not upload the whole folder manually to Colab.

Instead, generate a clean zip bundle:

```bash
python scripts/09_prepare_gpu_handoff.py --clear
```

This creates:

```text
outputs/gpu_handoff/
outputs/gpu_handoff.zip
```

Upload only:

```text
outputs/gpu_handoff.zip
```

Why use a zip:

- Colab upload is simpler,
- hidden local junk is excluded,
- the bundle includes only needed project artifacts,
- the manifest documents what was included,
- the extracted folder becomes a clean mini-repo.

The zip includes:

- `pyproject.toml`
- `uv.lock`
- `README.md`
- `configs/dpo_qwen15_lora.yaml`
- `data/tasks/`
- `data/preferences/`
- `data/trajectories/scored_train`
- `data/trajectories/scored_val`
- `data/trajectories/scored_test`
- `src/agentalign/`
- `scripts/`
- `tests/`
- `notebooks/02_colab_dpo_training.ipynb`
- `outputs/adapters/dpo_run/training_metadata.json`
- `manifest.json`
- `README_GPU_HANDOFF.md`

## 11. Local Handoff Verification

After building the handoff:

```bash
python scripts/08_audit_mvp.py
```

Look for:

```text
PASS colab_gpu_handoff_bundle
```

You can also inspect:

```bash
python -m json.tool outputs/gpu_handoff/manifest.json
```

The manifest should show counts like:

```text
train_preference_pairs: 336
val_preference_pairs: 9
scored_train_trajectories: 672
scored_val_trajectories: 31
scored_test_trajectories: 54
```

## 12. How To Run In Google Colab

Open:

```text
https://colab.research.google.com
```

### 12.1 Upload Notebook

In Colab:

```text
File -> Upload notebook
```

Upload:

```text
notebooks/02_colab_dpo_training.ipynb
```

This notebook is local in your repo.

### 12.2 Enable GPU

In Colab:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

Then:

```text
Runtime -> Restart session
```

Run:

```bash
!nvidia-smi
```

If you do not see an NVIDIA GPU, training will not work.

Colab GPU availability changes over time. The official Colab FAQ says GPU/TPU resources are not guaranteed and can vary by availability and usage limits.

### 12.3 Upload Project Bundle

The notebook has a cell:

```python
from google.colab import files
uploaded = files.upload()
```

When the upload dialog opens, upload:

```text
outputs/gpu_handoff.zip
```

Do not upload the full repo folder.

### 12.4 Extract Bundle

The notebook runs:

```bash
!rm -rf /content/agentalign_gpu
!mkdir -p /content/agentalign_gpu
!unzip -q gpu_handoff.zip -d /content/agentalign_gpu
%cd /content/agentalign_gpu
```

After this, Colab is inside:

```text
/content/agentalign_gpu
```

This is the extracted project root.

### 12.5 Install Dependencies In Colab

Run:

```bash
!python -m pip install -U pip
!python -m pip install -e ".[ml,dashboard]"
```

This installs:

- the local `agentalign` package,
- PyTorch,
- Transformers,
- TRL,
- PEFT,
- Datasets,
- bitsandbytes,
- Gradio/dashboard dependencies.

### 12.6 Run Preflight In Colab

Run:

```bash
!python scripts/05_train_dpo.py --preflight
!python scripts/08_audit_mvp.py
```

Expected:

- CUDA readiness should pass if GPU is active.
- adapter files will still fail before training.
- held-out adapter eval will still fail before eval.

### 12.7 Train Adapter In Colab

Run:

```bash
!python scripts/05_train_dpo.py \
  --train data/preferences/dpo_train.jsonl \
  --val data/preferences/dpo_val.jsonl \
  --output-dir outputs/adapters/qwen_dpo_final
```

This uses:

```text
configs/dpo_qwen15_lora.yaml
```

Expected adapter files:

```text
outputs/adapters/qwen_dpo_final/adapter_config.json
outputs/adapters/qwen_dpo_final/adapter_model.safetensors
```

### 12.8 Verify Adapter In Colab

Run:

```bash
!ls -lah outputs/adapters/qwen_dpo_final
!test -f outputs/adapters/qwen_dpo_final/adapter_config.json
!test -f outputs/adapters/qwen_dpo_final/adapter_model.safetensors
!python scripts/08_audit_mvp.py
```

At this stage:

- adapter artifacts should pass,
- held-out adapter eval may still fail until the next step.

### 12.9 Run Held-Out Base Vs Adapter Eval In Colab

Run base model on test tasks:

```bash
!python scripts/02_run_agent.py \
  --split test \
  --agent qwen_base \
  --model hf \
  --out runs/test_qwen \
  --clear
```

Run tuned adapter on same test tasks:

```bash
!python scripts/02_run_agent.py \
  --split test \
  --agent qwen_dpo \
  --model hf_adapter \
  --adapter-path outputs/adapters/qwen_dpo_final \
  --out runs/test_qwen \
  --clear
```

Score them:

```bash
!python scripts/03_score_trajectories.py \
  --runs-dir runs/test_qwen \
  --out-dir data/trajectories/scored_test_qwen \
  --clear
```

Compare:

```bash
!python scripts/06_eval_models.py \
  --runs-dir data/trajectories/scored_test_qwen \
  --split test \
  --agent qwen_dpo \
  --compare-agent qwen_base
```

Run final Colab audit:

```bash
!python scripts/08_audit_mvp.py
```

Full success requires:

- adapter files exist,
- base and adapter test trajectories exist,
- scored test eval exists,
- at least one held-out metric improves.

If no metric improves, the project can still be presented honestly as a rigorous eval/post-training system with a negative or mixed result.

## 13. Download Colab Results

The notebook zips:

```bash
!zip -qr qwen_dpo_final_artifacts.zip outputs/adapters/qwen_dpo_final data/trajectories/scored_test_qwen runs/test_qwen
files.download("qwen_dpo_final_artifacts.zip")
```

Download:

```text
qwen_dpo_final_artifacts.zip
```

This contains:

- trained adapter,
- raw base/tuned test runs,
- scored base/tuned test trajectories.

## 14. Bring Colab Results Back Locally

Back on your local machine:

```bash
cd /Users/shivashant/projects/AgentAlign-Lab
```

Extract the downloaded artifacts:

```bash
unzip /path/to/qwen_dpo_final_artifacts.zip -d .
```

Safer recommended import:

```bash
python scripts/10_import_colab_artifacts.py /path/to/qwen_dpo_final_artifacts.zip --clear
```

What this script does:

- checks the zip contains `adapter_config.json`,
- checks the zip contains `adapter_model.safetensors`,
- checks raw `runs/test_qwen` artifacts exist,
- checks scored `data/trajectories/scored_test_qwen` artifacts exist,
- extracts only the expected paths into the repo,
- prints a summary of imported files and trajectories.

Then run:

```bash
python scripts/08_audit_mvp.py
```

Expected after successful Colab training/eval:

```text
PASS dpo_adapter_artifacts
PASS dpo_held_out_adapter_eval
```

If local CUDA still fails, that is okay. The local Mac does not need to become a GPU machine.

## 15. How Each Major File Works

### `scripts/01_generate_tasks.py`

Creates task JSON files.

Interview explanation:

> I generate tasks programmatically so the suite is reproducible. Each task contains instructions, initial files, verifier configuration, and forbidden commands.

### `scripts/02_run_agent.py`

Runs one of four model modes:

- `dummy`: scripted success model,
- `bad_dummy`: scripted failure model,
- `hf`: base Hugging Face model,
- `hf_adapter`: base Hugging Face model plus LoRA adapter.

Interview explanation:

> I kept scripted agents for deterministic pipeline validation and added Hugging Face modes for real base-vs-adapter evaluation.

### `scripts/03_score_trajectories.py`

Loads saved runs, reruns/verifies them, and writes scored trajectories.

Interview explanation:

> I separate running from scoring so saved trajectories can be reprocessed and inspected independently.

### `scripts/04_build_preferences.py`

Creates DPO pairs from trajectories.

Interview explanation:

> For each task, I rank attempts by verifier score and create chosen/rejected pairs only when the margin is large enough. This avoids training on noisy near-ties.

### `scripts/05_train_dpo.py`

Does three jobs:

- `--dry-run`: validates data locally,
- `--preflight`: checks ML dependencies and CUDA,
- normal run: trains a QLoRA/DPO adapter.

Interview explanation:

> I made training preflight explicit because GPU dependency issues are common. Local dry-run validates the data without requiring CUDA.

### `scripts/06_eval_models.py`

Aggregates metrics:

- runs,
- pass rate,
- average score,
- invalid action rate,
- unsafe action rate,
- timeout rate,
- average steps.

Interview explanation:

> I report multiple behavior metrics because pass rate alone hides invalid actions, unsafe behavior, and inefficient loops.

### `scripts/08_audit_mvp.py`

Checks whether the project has the required artifacts.

Interview explanation:

> The audit script is a guardrail against vague claims. It checks task counts, split overlap, trajectory volume, preference pairs, handoff bundle contents, dashboard construction, adapter files, and held-out adapter evaluation.

### `scripts/09_prepare_gpu_handoff.py`

Creates the Colab upload bundle.

Interview explanation:

> Since my local machine may not have CUDA, I package the exact data, code, config, notebook, and audit evidence into a clean Colab handoff zip.

### `scripts/10_import_colab_artifacts.py`

Validates and imports the artifact zip downloaded from Colab.

Interview explanation:

> I added an import script so the Colab-to-local handoff is reproducible. It validates that the adapter and held-out eval artifacts exist before extracting them into the expected local paths.

## 16. Metrics Explained

### Pass Rate

Percentage of runs where verifier passed.

Good for:

- overall correctness.

Weakness:

- hides how the agent behaved before success/failure.

### Average Score

Composite verifier score.

Current scoring idea:

```text
score = success reward
      - step penalty
      - failed command penalty
      - invalid action penalty
      - unsafe action penalty
```

Good for:

- ranking trajectories,
- building preference pairs.

### Invalid Action Rate

How often the model outputs actions that cannot be parsed or are not allowed.

Good for:

- measuring tool-use reliability,
- detecting format degradation after tuning.

### Unsafe Action Rate

How often the model attempts forbidden behavior.

Good for:

- safety evaluation,
- checking whether DPO reduces bad tool use.

### Timeout Rate

How often the agent fails to finish in max steps/time.

Good for:

- detecting loops and indecision.

### Average Steps

How many actions the agent takes.

Good for:

- efficiency,
- comparing whether tuning solves tasks faster.

## 17. Evaluation Integrity

The biggest interview risk is evaluation leakage.

The project avoids this by:

- splitting tasks by task ID,
- building train preferences only from train tasks,
- keeping test tasks held out,
- comparing base and tuned models on the same test split,
- using the same tools, prompts, max steps, and verifiers.

Say:

> I split by task before trajectory generation, not after, so the preference dataset cannot contain attempts from held-out tasks.

Do not say:

> I randomly split trajectories.

That would be weaker because trajectories from the same task could leak across train/test.

## 18. Common Interview Questions

### Q: Why deterministic verifiers instead of an LLM judge?

Answer:

> For this MVP, deterministic verifiers are cheaper, reproducible, and less noisy. Since the tasks are small terminal tasks, correctness can often be checked with tests, exact files, JSON schema, or safety-aware checks. LLM judges are useful future work for semantic tasks, but they would add judge bias and variance here.

### Q: Why DPO instead of PPO/RLHF?

Answer:

> DPO is simpler and more stable for an MVP. It uses offline preference pairs and avoids training a separate reward model or running online RL. Since my feedback source is verifier-ranked trajectories, DPO is a good fit.

### Q: Why LoRA/QLoRA?

Answer:

> Full fine-tuning is too expensive for a student/free-GPU project. LoRA trains small adapter matrices while freezing the base model. QLoRA reduces memory pressure further with quantization, making Colab training more realistic.

### Q: What proves the preferences are correct?

Answer:

> Each pair is created from same-task trajectories, and the chosen trajectory has a higher verifier score than the rejected one by a minimum margin. The data is schema-validated, and the audit checks pair counts and split separation. This does not make labels perfect, but it makes them reproducible and inspectable.

### Q: What if DPO does not improve pass rate?

Answer:

> That is a valid result. The project tracks multiple metrics, including invalid actions, unsafe actions, timeout rate, average steps, and verifier score. If pass rate does not improve, I would inspect whether format reliability or safety improved, and report the limitation honestly.

### Q: Why use scripted baselines?

Answer:

> Scripted baselines are for deterministic pipeline validation. They prove that task generation, trajectory logging, scoring, preference building, evaluation, and dashboard loading work before spending GPU time. The real model comparison is base Qwen vs DPO adapter on held-out tasks.

### Q: What are the biggest limitations?

Answer:

> The tasks are synthetic and compact, deterministic verifiers can be gamed, scripted data proves pipeline mechanics but not model generalization, and the small model may not reflect frontier-agent behavior. I mitigate this with held-out splits, safety checks, failure analysis, and honest reporting.

### Q: What would you build next?

Answer:

> The highest-value next step is Dockerized reproducible evaluation and richer trace export, because isolation and observability become more important as tasks become more realistic. After that, I would explore step-level preferences or AI-assisted failure tagging.

## 19. Debugging Guide

Debug by layer:

```text
schema -> data -> runner -> parser -> tool -> verifier -> preference -> training -> eval -> dashboard
```

### If Task Generation Fails

Check:

```bash
python scripts/01_generate_tasks.py
```

Then inspect:

```bash
ls data/tasks
python -m json.tool data/tasks/py_fix_001.json
```

### If Agent Runs Fail

Run one task:

```bash
python scripts/02_run_agent.py --task-id py_fix_001 --agent debug --model dummy --out runs/debug --clear
```

Inspect:

```bash
ls runs/debug
```

### If Scoring Fails

Run:

```bash
python scripts/03_score_trajectories.py --runs-dir runs/debug --out-dir data/trajectories/debug_scored --clear
```

### If DPO Dry-Run Fails

Check first line:

```bash
head -1 data/preferences/dpo_train.jsonl
```

Validate JSON:

```bash
python -m json.tool data/preferences/dpo_train.jsonl
```

Note: `json.tool` expects one JSON document, so it may not work on full JSONL. If needed, inspect one line only.

### If Colab Training Fails

First check GPU:

```bash
!nvidia-smi
```

Then preflight:

```bash
!python scripts/05_train_dpo.py --preflight
```

If CUDA OOM:

- reduce `max_seq_length`,
- keep `per_device_train_batch_size: 1`,
- reduce epochs,
- restart runtime and rerun.

### If Adapter Eval Fails

Run one test task:

```bash
!python scripts/02_run_agent.py --task-id py_fix_001 --agent qwen_dpo_debug --model hf_adapter --adapter-path outputs/adapters/qwen_dpo_final --out runs/debug_qwen --clear
```

Then score it:

```bash
!python scripts/03_score_trajectories.py --runs-dir runs/debug_qwen --out-dir data/trajectories/debug_qwen --clear
```

## 20. What To Memorize For Interviews

Memorize these concise answers.

### Project Pitch

> AgentAlign Lab is a verifier-guided post-training pipeline for terminal agents. It logs agent trajectories, scores them with deterministic verifiers, converts high-margin successes/failures into DPO pairs, trains a QLoRA adapter, and evaluates base vs tuned behavior on held-out tasks.

### Why It Matters

> Modern agent systems need reliable feedback loops. This project demonstrates the loop: run, trace, verify, curate data, post-train, evaluate, and inspect failures.

### Main Engineering Tradeoff

> I chose small deterministic tasks and local JSONL artifacts over a large benchmark because the goal was a complete, reproducible feedback loop rather than broad benchmark claims.

### Main ML Tradeoff

> I used DPO/LoRA because it is cheaper and more stable than PPO-style RLHF for a student-scale project.

### Main Evaluation Claim

> I split by task ID before generating preferences, then evaluate base and tuned models on held-out test tasks with the same tools, prompts, and verifiers.

### Main Limitation

> The task suite is compact and synthetic, so results are local evidence about this feedback loop, not universal claims about coding agents.

## 21. Final Success Criteria

The full project is truly complete when all are true:

- 60+ verified tasks exist.
- 300+ trajectories exist.
- 300+ preference pairs exist.
- DPO training completes on CUDA.
- Adapter files exist.
- Held-out base-vs-adapter eval runs.
- At least one held-out metric improves.
- Dashboard loads real artifacts.
- README/report honestly describe methods, results, and limitations.
- Audit confirms artifacts.

Current local state can prove most of this. Colab must prove:

- actual GPU training,
- adapter files,
- held-out adapter eval,
- metric delta.

## 22. Final Local Commands After Colab

After bringing Colab artifacts back:

```bash
cd /Users/shivashant/projects/AgentAlign-Lab
python scripts/08_audit_mvp.py
python scripts/06_eval_models.py --runs-dir data/trajectories/scored_test_qwen --split test --agent qwen_dpo --compare-agent qwen_base
python scripts/07_launch_dashboard.py
```

Then open the dashboard URL printed by the launcher.

Use the dashboard to inspect:

- overview metrics,
- test trajectories,
- qwen_base vs qwen_dpo comparison,
- failure modes,
- preference pairs.

## 23. Study Schedule

If you have 1 day:

1. Read Sections 1-6 of this guide.
2. Run local audit.
3. Generate Colab bundle.
4. Skim `scripts/05_train_dpo.py`.
5. Practice the project pitch and common questions.

If you have 3 days:

Day 1:

- Read this guide through Section 11.
- Run all local commands.
- Understand schemas, agent loop, verifier.

Day 2:

- Read DPO/LoRA/QLoRA resources.
- Run Colab training.
- Bring artifacts back.

Day 3:

- Study dashboard/report.
- Practice interview answers.
- Prepare a 90-second demo.

If you have 1 week:

- Day 1: tasks, schemas, verifiers.
- Day 2: agent loop, tools, trajectories.
- Day 3: preferences and DPO data.
- Day 4: Colab training.
- Day 5: evaluation and dashboard.
- Day 6: report and limitations.
- Day 7: mock interviews and demo rehearsal.

## 24. What To Skip For Now

Skip these until the core project is done:

- PPO/RLHF implementation,
- reward model training,
- GRPO,
- multi-agent systems,
- browser automation,
- Kubernetes,
- production observability tools,
- large benchmark replication,
- fancy UI polish.

These are useful future-work topics, not MVP requirements.

## 25. Final Interview Framing

A strong final explanation:

> I built AgentAlign Lab to understand the full feedback loop behind reliable terminal agents. The system generates deterministic tasks, runs an agent with structured JSON tools, logs every action, scores outcomes with verifiers, creates high-margin preference pairs, trains a LoRA adapter with DPO on Colab, and evaluates base vs tuned behavior on held-out tasks. I focused on reproducibility and evaluation integrity: task-level splits, deterministic verifiers, JSONL artifacts, audit scripts, and honest limitations. The result is a small but complete post-training and observability pipeline.

If the model improves:

> The tuned model improved on [metric], while [other metrics] changed by [delta]. I interpret this as local evidence that verifier-guided preference data can improve this narrow terminal-agent distribution.

If the model does not improve:

> The training result was mixed, but the project still produced a rigorous agent evaluation and trace-analysis system. I would next inspect pair quality, prompt compatibility, and step-level failure modes before doing more tuning.
