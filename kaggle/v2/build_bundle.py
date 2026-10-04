#!/usr/bin/env python3
"""Build the Kaggle v2 bundle: the dataset folder/zip and the notebook.

    python kaggle/v2/build_bundle.py

Writes:
  kaggle/v2/dataset/agentalign-v2/         project snapshot to upload as a Kaggle Dataset
  kaggle/v2/agentalign-v2-dataset.zip      the same folder, zipped (upload either one)
  kaggle/v2/agentalign_v2_kaggle.ipynb     the notebook to import into Kaggle
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASET = HERE / "dataset" / "agentalign-v2"
ZIP = HERE / "agentalign-v2-dataset"

INCLUDE = ["src", "scripts", "tests", "configs", "data/tasks", "pyproject.toml", "README.md"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".DS_Store")

# Versions tested together locally (torch is taken from the Kaggle image).
PINS = [
    "transformers==5.9.0",
    "trl==1.6.0",
    "peft==0.19.1",
    "datasets==5.0.0",
    "accelerate==1.14.0",
    "bitsandbytes==0.49.2",
]


def build_dataset() -> None:
    if DATASET.exists():
        shutil.rmtree(DATASET)
    DATASET.mkdir(parents=True)
    for rel in INCLUDE:
        src = ROOT / rel
        dst = DATASET / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    (DATASET / "AGENTALIGN_V2_DATASET").write_text("AgentAlign-Lab v2 project snapshot\n")
    (DATASET / "dataset-metadata.json").write_text(json.dumps({
        "title": "agentalign-v2",
        "id": "YOUR_KAGGLE_USERNAME/agentalign-v2",
        "licenses": [{"name": "MIT"}],
    }, indent=2))
    shutil.make_archive(str(ZIP), "zip", DATASET)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# AgentAlign-Lab v2: on-policy RFT + step-level DPO

**Kaggle settings (right-hand panel):** Accelerator **GPU T4 x2** · Internet **On** · Input: add the
`agentalign-v2` dataset. Then use **Save Version → Save & Run All (Commit)** so the run keeps going with the
browser closed (up to 12 h). Expected total: about 4–6 h.

What it does:
1. **Smoke test** (about 10–15 min): every stage on 2–3 tasks, so an environment problem fails in minutes, not hours.
2. **Rollouts:** the base model tries each of the 81 train tasks 16 times (batched, both GPUs), logging the exact prompt and output of every step.
3. **Data:** clean steps from successful runs become SFT examples. At those same states, alternative actions are sampled and kept as DPO "rejected" only when a deterministic check proves them bad (invalid JSON, disallowed command, editing tests, unsafe path, or `final_answer` while the verifier still fails).
4. **Training:** SFT (QLoRA) on successful steps, merged, then DPO (QLoRA) on top. Whole tasks are held out for eval loss.
5. **Evaluation:** base vs SFT vs SFT+DPO on the 42 held-out test tasks × 4 runs, same harness and settings as v1, with paired bootstrap CIs.
6. **Export:** `/kaggle/working/agentalign_v2_results.zip` (download it from the Output tab).

If the session stops early, attach that results zip as an extra input and run again: finished stages are skipped.
"""),
    code(r"""
# 1. Settings
import json, os, shutil, subprocess, sys, time
from pathlib import Path

NOTEBOOK_START = time.time()
MODEL = os.environ.get("AGENTALIGN_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
INPUT_ROOT = Path(os.environ.get("AGENTALIGN_INPUT", "/kaggle/input"))
WORK_ROOT = Path(os.environ.get("AGENTALIGN_WORK", "/kaggle/working"))
RUN_SMOKE_FIRST = os.environ.get("AGENTALIGN_SMOKE_FIRST", "1") == "1"
RUN_FULL = os.environ.get("AGENTALIGN_RUN_FULL", "1") == "1"
# Hard stops (hours after the notebook starts) so evaluation always has time to finish.
ROLLOUT_DEADLINE_H = 5.0
ALTS_DEADLINE_H = 5.75

LORA = dict(lora_r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
FULL = dict(
    train_samples=16, rollout_temperature=0.9, rollout_top_p=0.95, max_steps=8, max_new_tokens=512,
    alt_k=4, alt_temperature=1.0, max_success_per_task=4, max_pairs_per_state=2, max_dpo_pairs=1200,
    eval_reps=4, eval_temperature=0.8, max_tasks=None, eval_max_tasks=None, smoke=False,
    sft=dict(epochs=3, lr=1e-4, batch_size=2, grad_accum=8, max_length=3072, warmup_ratio=0.1,
             logging_steps=5, eval_steps=20, quantize=True, **LORA),
    dpo=dict(epochs=2, lr=2e-5, beta=0.1, batch_size=1, grad_accum=16, max_length=3072, warmup_ratio=0.1,
             logging_steps=5, eval_steps=20, quantize=True, **LORA),
)
SMOKE = dict(
    FULL, train_samples=2, max_steps=3, max_new_tokens=128, alt_k=2, max_dpo_pairs=16, eval_reps=1,
    max_tasks=3, eval_max_tasks=2, smoke=True,
    sft=dict(FULL["sft"], max_steps=2, max_length=2048, eval_steps=1),
    dpo=dict(FULL["dpo"], max_steps=2, max_length=2048, eval_steps=1),
)
# Local testing only: shrink the full run, e.g. AGENTALIGN_FULL_OVERRIDES='{"max_tasks": 2}'
FULL.update(json.loads(os.environ.get("AGENTALIGN_FULL_OVERRIDES", "{}")))
print("model:", MODEL, "| smoke first:", RUN_SMOKE_FIRST, "| full run:", RUN_FULL)
"""),
    code(r"""
# 2. Install the tested library versions (torch comes from the Kaggle image)
PINS = __PINS__
if os.environ.get("AGENTALIGN_SKIP_PIP") != "1":
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS, "jsonlines>=4.0"],
                       capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-4000:])
    r.check_returncode()
    # Kaggle ships torchao 0.10, which peft 0.19 rejects when loading LoRA adapters. Nothing here uses torchao.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], capture_output=True, text=True)
import importlib.util
print("torchao present:", importlib.util.find_spec("torchao") is not None, "(the code also guards against an old torchao)")
import torch
N_GPU = torch.cuda.device_count()
print("torch", torch.__version__, "| GPUs:", N_GPU, [torch.cuda.get_device_name(i) for i in range(N_GPU)])
if N_GPU == 0 and os.environ.get("AGENTALIGN_ALLOW_CPU") != "1":
    raise SystemExit("No GPU. In the right panel set Accelerator = GPU T4 x2, then run again.")
GPUS = list(range(N_GPU)) if N_GPU else [None, None]
"""),
    code(r"""
# 3. Find the attached dataset and set up the project
def find_marker(name):
    hits = sorted(INPUT_ROOT.rglob(name)) if INPUT_ROOT.exists() else []
    if hits:
        return hits[0].parent
    for z in sorted(INPUT_ROOT.rglob("*.zip")) if INPUT_ROOT.exists() else []:
        out = WORK_ROOT / "_unzipped" / z.stem
        if not out.exists():
            shutil.unpack_archive(str(z), str(out))
        hits = sorted(out.rglob(name))
        if hits:
            return hits[0].parent
    return None

SOURCE = find_marker("AGENTALIGN_V2_DATASET")
if SOURCE is None:
    raise SystemExit("Dataset not found. Add the 'agentalign-v2' dataset under Input, then run again.")
print("dataset:", SOURCE)

def setup_project(dest):
    dest.mkdir(parents=True, exist_ok=True)
    for item in SOURCE.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    return dest

PROJECT = setup_project(WORK_ROOT / "AgentAlign-Lab")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT)], capture_output=True, text=True)
print(r.stdout[-1000:], r.stderr[-2000:]); r.check_returncode()
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_v2.py"],
                   cwd=PROJECT, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:]); r.check_returncode()

# Download the model once, before several processes need it.
from huggingface_hub import snapshot_download
snapshot_download(MODEL)
print("model cached")
"""),
    code(r"""
# 4. Resume: if a previous results zip/folder is attached, restore its finished stages
PREVIOUS = find_marker("AGENTALIGN_V2_RESULTS")
if PREVIOUS is not None and not (PROJECT / "v2" / ".done_report").exists():
    shutil.copytree(PREVIOUS, PROJECT / "v2", dirs_exist_ok=True)
    print("restored previous results from", PREVIOUS)
else:
    print("no previous results attached (fresh run)")
"""),
    code(r"""
# 5. Stage runner: one subprocess per GPU, live progress, full log tail on failure
def _env(gpu):
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env

def run_parallel(cmds, gpus, log_prefix, cwd):
    procs = []
    for i, (cmd, gpu) in enumerate(zip(cmds, gpus)):
        path = Path(f"{log_prefix}_{i}.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = path.open("w")
        procs.append((subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=_env(gpu), stdout=fh,
                                       stderr=subprocess.STDOUT, text=True), fh, path, gpu))
    seen = [0] * len(procs)
    while True:
        running = False
        for i, (p, fh, path, gpu) in enumerate(procs):
            text = path.read_text(errors="replace")
            for line in text[seen[i]:].splitlines():
                s = line.strip()
                if s.startswith("[") or s.startswith("{'loss'") or s.startswith("{'eval_loss'") or s.startswith("{'train_runtime'"):
                    print(f"[gpu {gpu}] {s[:300]}", flush=True)
            seen[i] = len(text)
            running = running or p.poll() is None
        if not running:
            break
        time.sleep(10)
    failed = []
    for p, fh, path, gpu in procs:
        fh.close()
        if p.returncode != 0:
            failed.append(path)
    for path in failed:
        print(f"\n===== FAILED: {path} (last 80 lines) =====")
        print("\n".join(path.read_text(errors="replace").splitlines()[-80:]))
    if failed:
        raise RuntimeError(f"stage failed: {[str(p) for p in failed]}")

def count_jsonl(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("*.jsonl")) if path.exists() else []
    return sum(1 for f in files for line in f.read_text().splitlines() if line.strip())
"""),
    code(r"""
# 6. The pipeline
def pipeline(P, cfg, label):
    V2 = P / "v2"
    for d in ["logs", "data", "adapters", "models", "eval", "figures"]:
        (V2 / d).mkdir(parents=True, exist_ok=True)
    py = [sys.executable, "scripts/v2_pipeline.py"]
    n = len(GPUS)
    gen = ["--model", MODEL, "--max-new-tokens", cfg["max_new_tokens"]]
    def mark(stage):
        (V2 / f".done_{stage}").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    def done(stage):
        return (V2 / f".done_{stage}").exists()
    def t():
        return f"{(time.time() - NOTEBOOK_START) / 3600:.2f} h"
    max_tasks = ["--max-tasks", cfg["max_tasks"]] if cfg["max_tasks"] else []

    if not done("rollouts"):
        print(f"\n## [{label}] rollouts on train ({t()})")
        deadline = NOTEBOOK_START + ROLLOUT_DEADLINE_H * 3600
        run_parallel([py + ["rollout", "--split", "train", "--samples", cfg["train_samples"], "--agent-id", "qwen_base_sample",
                            "--out", V2 / "rollouts_train", "--temperature", cfg["rollout_temperature"],
                            "--top-p", cfg["rollout_top_p"], "--max-steps", cfg["max_steps"], "--shard", i,
                            "--nshards", n, "--deadline", deadline] + gen + max_tasks for i in range(n)],
                     GPUS, V2 / "logs" / "rollouts", P)
        mark("rollouts")
    print("train rollouts:", count_jsonl(V2 / "rollouts_train"))

    if not done("states"):
        run_parallel([py + ["states", "--rollouts", V2 / "rollouts_train", "--out", V2 / "data" / "states.jsonl",
                            "--max-success-per-task", cfg["max_success_per_task"]]], [GPUS[0]], V2 / "logs" / "states", P)
        mark("states")
    print((V2 / "data" / "states.summary.json").read_text())

    if not done("alts"):
        if count_jsonl(V2 / "data" / "states.jsonl"):
            print(f"\n## [{label}] alternative actions ({t()})")
            deadline = NOTEBOOK_START + ALTS_DEADLINE_H * 3600
            run_parallel([py + ["alts", "--states", V2 / "data" / "states.jsonl", "--k", cfg["alt_k"],
                                "--temperature", cfg["alt_temperature"], "--out", V2 / "alts", "--shard", i,
                                "--nshards", n, "--deadline", deadline] + gen for i in range(n)],
                         GPUS, V2 / "logs" / "alts", P)
        mark("alts")

    if not done("build"):
        cmd = py + ["build", "--states", V2 / "data" / "states.jsonl", "--alts", V2 / "alts", "--out-dir", V2 / "data",
                    "--max-pairs-per-state", cfg["max_pairs_per_state"], "--max-dpo-pairs", cfg["max_dpo_pairs"]]
        if cfg["smoke"]:
            cmd += ["--smoke-fallback", "--rollouts", V2 / "rollouts_train"]
        run_parallel([cmd], [GPUS[0]], V2 / "logs" / "build", P)
        mark("build")
    print((V2 / "data" / "build_summary.json").read_text())

    sft_merged = V2 / "models" / "sft_merged"
    if not done("sft"):
        if count_jsonl(V2 / "data" / "sft_train.jsonl"):
            print(f"\n## [{label}] SFT ({t()})")
            cfg_path = V2 / "data" / "sft_config.json"
            cfg_path.write_text(json.dumps(cfg["sft"]))
            ev = ["--eval", V2 / "data" / "sft_eval.jsonl"] if count_jsonl(V2 / "data" / "sft_eval.jsonl") else []
            run_parallel([py + ["train", "--kind", "sft", "--model", MODEL, "--train", V2 / "data" / "sft_train.jsonl",
                                "--config", cfg_path, "--out", V2 / "adapters" / "sft"] + ev], [GPUS[0]], V2 / "logs" / "sft", P)
        else:
            print("No successful training steps: skipping SFT.")
        mark("sft")
    # Merge outside the SFT stage: a resumed run restores the adapter but not the 3 GB merged model.
    if (V2 / "adapters" / "sft" / "adapter_config.json").exists() and not (sft_merged / "config.json").exists():
        run_parallel([py + ["merge", "--base", MODEL, "--adapter", V2 / "adapters" / "sft", "--out", sft_merged]],
                     [GPUS[0]], V2 / "logs" / "merge", P)
    dpo_base = sft_merged if (sft_merged / "config.json").exists() else MODEL

    if not done("dpo"):
        if count_jsonl(V2 / "data" / "dpo_train.jsonl"):
            print(f"\n## [{label}] DPO on {dpo_base} ({t()})")
            cfg_path = V2 / "data" / "dpo_config.json"
            cfg_path.write_text(json.dumps(cfg["dpo"]))
            ev = ["--eval", V2 / "data" / "dpo_eval.jsonl"] if count_jsonl(V2 / "data" / "dpo_eval.jsonl") else []
            run_parallel([py + ["train", "--kind", "dpo", "--model", dpo_base, "--train", V2 / "data" / "dpo_train.jsonl",
                                "--config", cfg_path, "--out", V2 / "adapters" / "dpo"] + ev], [GPUS[0]], V2 / "logs" / "dpo", P)
        else:
            print("No DPO pairs: skipping DPO.")
        mark("dpo")

    arms = [("qwen_base", MODEL, None)]
    if (sft_merged / "config.json").exists():
        arms.append(("qwen_sft", sft_merged, None))
    if (V2 / "adapters" / "dpo" / "adapter_config.json").exists():
        arms.append(("qwen_sft_dpo" if dpo_base == sft_merged else "qwen_dpo", dpo_base, V2 / "adapters" / "dpo"))
    eval_tasks = ["--max-tasks", cfg["eval_max_tasks"]] if cfg["eval_max_tasks"] else []
    for arm, model, adapter in arms:
        if done(f"eval_{arm}"):
            continue
        print(f"\n## [{label}] evaluate {arm} on test ({t()})")
        extra = ["--adapter", adapter] if adapter else []
        run_parallel([py + ["rollout", "--split", "test", "--samples", cfg["eval_reps"], "--agent-id", arm,
                            "--out", V2 / "eval" / arm, "--model", model, "--max-new-tokens", cfg["max_new_tokens"],
                            "--temperature", cfg["eval_temperature"], "--top-p", 1.0, "--max-steps", cfg["max_steps"],
                            "--shard", i, "--nshards", n] + extra + eval_tasks for i in range(n)],
                     GPUS, V2 / "logs" / f"eval_{arm}", P)
        mark(f"eval_{arm}")

    run_parallel([py + ["report", "--eval-dir", V2 / "eval", "--arms", ",".join(a for a, _, _ in arms),
                        "--out", V2 / "results" / "eval_results.json"]], [GPUS[0]], V2 / "logs" / "report", P)
    mark("report")
    print((V2 / "results" / "eval_results.md").read_text())
    print(f"[{label}] finished at {t()}")
    return V2
"""),
    code(r"""
# 7. Smoke test: every stage on a few tasks (about 10-15 min)
if RUN_SMOKE_FIRST:
    smoke_project = setup_project(WORK_ROOT / "smoke" / "AgentAlign-Lab")
    smoke_v2 = pipeline(smoke_project, SMOKE, "smoke")
    # Keep the smoke report and logs (small); drop its models and rollouts.
    shutil.rmtree(WORK_ROOT / "smoke_report", ignore_errors=True)
    for sub in ["results", "logs", "data"]:
        if (smoke_v2 / sub).exists():
            shutil.copytree(smoke_v2 / sub, WORK_ROOT / "smoke_report" / sub)
    shutil.rmtree(WORK_ROOT / "smoke", ignore_errors=True)
    print("\nSMOKE TEST PASSED - starting the full run")
"""),
    code(r"""
# 8. Full run (about 4-6 h on T4 x2)
if RUN_FULL:
    V2 = pipeline(PROJECT, FULL, "full")
"""),
    code(r"""
# 9. Figures: training curves and pass rates with 95% CIs
import matplotlib.pyplot as plt

V2 = PROJECT / "v2"
for kind, keys in [("sft", ["loss", "eval_loss"]), ("dpo", ["loss", "eval_loss", "rewards/accuracies", "rewards/margins"])]:
    meta_path = V2 / "adapters" / kind / "v2_training_meta.json"
    if not meta_path.exists():
        continue
    hist = json.loads(meta_path.read_text())["log_history"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3))
    for ax, key in zip(axes, keys):
        pts = [(h["step"], h[key]) for h in hist if key in h and "step" in h]
        if pts:
            ax.plot(*zip(*pts), marker=".")
        ax.set_title(f"{kind}: {key}")
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(V2 / "figures" / f"{kind}_curves.png", dpi=120)
    plt.show()

res_path = V2 / "results" / "eval_results.json"
if res_path.exists():
    res = json.loads(res_path.read_text())
    arms = list(res["arms"])
    rates = [res["arms"][a]["pass_rate"] * 100 for a in arms]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(arms, rates, color=["#8c8c8c", "#4c78a8", "#f58518", "#54a24b"][:len(arms)])
    for i, r in enumerate(rates):
        ax.text(i, r + 0.3, f"{r:.1f}%", ha="center")
    ax.set_ylabel("pass rate on held-out test (%)")
    ax.set_title("AgentAlign v2: test pass rate")
    fig.tight_layout()
    fig.savefig(V2 / "figures" / "pass_rate.png", dpi=120)
    plt.show()
"""),
    code(r"""
# 10. Export everything except the merged 3 GB model (adapters are enough to rebuild it)
V2 = PROJECT / "v2"
if not (V2 / ".done_report").exists():
    print("The full run has not finished, so there is nothing to export yet.")
else:
    (V2 / "AGENTALIGN_V2_RESULTS").write_text("AgentAlign v2 results\n")
    export = WORK_ROOT / "agentalign_v2_results"
    shutil.rmtree(export, ignore_errors=True)
    shutil.copytree(V2, export, ignore=shutil.ignore_patterns("models"))
    archive = shutil.make_archive(str(WORK_ROOT / "agentalign_v2_results"), "zip", export)
    shutil.rmtree(export, ignore_errors=True)
    print("download from the Output tab:", archive, f"({Path(archive).stat().st_size / 2**20:.1f} MB)")
print(f"total time: {(time.time() - NOTEBOOK_START) / 3600:.2f} h")
"""),
]


def build_notebook() -> None:
    cells = json.loads(json.dumps(CELLS))
    for cell in cells:
        cell["source"] = [line.replace("__PINS__", json.dumps(PINS)) for line in cell["source"]]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / "agentalign_v2_kaggle.ipynb").write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    build_dataset()
    build_notebook()
    print("dataset:", DATASET)
    print("zip:    ", ZIP.with_suffix(".zip"))
    print("notebook:", HERE / "agentalign_v2_kaggle.ipynb")
