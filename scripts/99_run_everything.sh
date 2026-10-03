#!/bin/bash
set -e

echo "=== STEP 1: Real LLM Rollouts ==="
# Train split: 42 tasks * 8 reps = 336 trajectories (~2.5 hours)
python scripts/11_run_llm_rollouts.py --split train --reps 8
# Val split: 10 tasks * 2 reps = 20 trajectories (~10 mins)
python scripts/11_run_llm_rollouts.py --split val --reps 2

echo "=== STEP 2: DPO Training (Local MPS) ==="
# Training should take about 15-30 minutes locally
python scripts/12_colab_train_and_eval.py

echo "=== STEP 3: Evaluation ==="
# Eval on 8 test tasks * 4 reps * 2 models = 64 trajectories (~30 mins)
python scripts/13_eval_base_vs_tuned.py --split test --reps 4

echo "=== STEP 4: Export Final Model ==="
python scripts/14_merge_and_export.py

echo "=== ALL DONE! ==="
