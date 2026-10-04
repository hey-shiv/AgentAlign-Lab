# AgentAlign v2 on Kaggle

## Why v2
v1 ended in a null result (DPO 8.4% vs base 7.7%, CI spanning zero). Two causes:
1. **Training data did not match inference.** v1 pairs used the prompt `"Task: <id>"` and whole trajectories
   (including environment observations) as completions. At inference the agent sees the full system prompt,
   history and chat template, and writes one JSON action. v2 trains on exactly that per-step prompt and output.
2. **Too few successes.** Only 11 of 80 v1 "chosen" trajectories succeeded. v2 samples 16 runs per train task
   (batched across both GPUs), keeps clean successful steps for SFT, and builds step-level DPO pairs whose
   "rejected" side is proven bad by a deterministic check.

## Run it
1. Build the bundle (already built if you got these files from me):
   `python kaggle/v2/build_bundle.py`
2. Kaggle → **Datasets → New Dataset** → upload `kaggle/v2/agentalign-v2-dataset.zip`, name it `agentalign-v2`.
3. Kaggle → **Code → New Notebook → File → Import Notebook** → `kaggle/v2/agentalign_v2_kaggle.ipynb`.
4. Right panel: **Accelerator: GPU T4 x2**, **Internet: On** (needs a phone-verified account),
   **Add Input** → your `agentalign-v2` dataset.
5. **Save Version → Save & Run All (Commit)**. You can close the browser.
6. When it finishes, open the version's **Output** tab and download `agentalign_v2_results.zip`.
   Unzip it into `AgentAlign-Lab/v2/` locally.

The notebook first runs a smoke test of every stage on 2–3 tasks (about 10–15 min) and only then starts the
full run, so environment problems fail early.

**If the run stops early:** create a new version with `agentalign_v2_results.zip` (from the stopped run's Output)
added as a second input. Finished stages are skipped and unfinished rollouts resume where they stopped.

## Stages and expected time on T4 x2
| Stage | Work | Time |
|---|---|---|
| Setup + smoke test | install, model download, every stage on 2–3 tasks | 15–25 min |
| Train rollouts | 81 tasks × 16 samples, batched on 2 GPUs | 45–75 min |
| Alternatives | 4 sampled actions per state | 10–15 min |
| SFT (QLoRA) | successful steps, 3 epochs | 20–35 min |
| Merge | SFT adapter into fp16 model | 2–5 min |
| DPO (QLoRA) | up to 1,200 step pairs, 2 epochs | 35–70 min |
| Evaluation | base, SFT, SFT+DPO × 42 test tasks × 4 runs | 30–50 min |
| **Total** | | **about 3–5 h** (hard stops keep it under 12 h) |

## Outputs (`v2/`)
- `results/eval_results.json` and `.md`: per-arm metrics and paired bootstrap 95% CIs
- `figures/`: training curves and pass-rate chart
- `data/`: states, SFT examples, DPO pairs, `build_summary.json` (reject reasons, held-out tasks)
- `adapters/sft`, `adapters/dpo`: LoRA adapters with `v2_training_meta.json` (mode, time, GPU, memory, log history)
- `rollouts_train/`, `eval/`: every trajectory with the exact prompt and completion of each step
- `logs/`: one log per stage and GPU

## Local check
`python -m pytest tests/test_v2.py` runs the v2 unit tests on CPU. The whole notebook was also executed
end-to-end locally in smoke mode (CPU, two shards, small Qwen model) before release.
