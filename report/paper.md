# AgentAlign Lab: Technical Report

**Title:** Verifier-Guided Preference Data for Reliable Terminal Agents  
**Author:** Shivashant Manohar

## Abstract

Terminal agents often look confident while failing tests, producing invalid tool calls, or taking unsafe actions. AgentAlign Lab implements a compact local feedback loop: deterministic terminal tasks, structured agent traces, verifier scoring, high-margin preference-pair generation, split-aware evaluation, and a Gradio dashboard. A first on-policy run (v1) produced no detectable improvement. We traced that null to a mismatch between training and inference inputs and to a shortage of successful rollouts, and fixed both in v2. With supervised fine-tuning on clean successful steps followed by DPO on step-level, verifier-labelled pairs, `Qwen/Qwen2.5-Coder-1.5B-Instruct` improves from 7.1% to 11.3% pass rate on 42 unseen tasks (+4.2 pts, paired bootstrap 95% CI [+1.2, +7.7]) and the share of runs that try a disallowed command falls from 41.1% to 27.4% (-13.7 pts, CI [-24.4, -3.0]).

## 1. Introduction

Modern agent systems need tight feedback loops: run the agent, trace each action, verify the outcome, create training/evaluation data, and inspect regressions. This project is intentionally narrow: it is not a SWE-bench clone or production observability platform. It is a small, reproducible slice of the same systems problem.

Primary question:

Can deterministic verifier feedback on an agent's own rollouts produce useful on-policy preference data for improving terminal-agent reliability via DPO?

## 2. Task Suite

The implemented task suite contains 140 verified tasks across the handbook MVP families, completely split by task template to prevent test leakage:

| Family | Count | Verifier |
|---|---:|---|
| Python bug fix | 50 | `pytest` |
| Data transformation | 30 | exact JSON |
| JSON/config repair | 25 | JSON parser/schema |
| Log extraction | 20 | exact file |
| Safety trap | 15 | safety-aware `pytest` |

Tasks are split into Train (81), Validation (17), and Test (42) splits. 

## 3. Agent And Sandbox

The agent loop uses a custom single-agent implementation with strict JSON actions:

- `list_files`
- `read_file`
- `write_file`
- `run_command`
- `final_answer`

Each task runs in a Python temporary directory. Tool execution uses `subprocess.run` without `shell=True`, enforces timeouts, restricts commands to an allowlist, and prevents file access outside the workspace.

## 4. Verifier And Scoring

Verifier results are deterministic and family-specific. The composite score follows the handbook shape:

```text
score = 10.0 * success
      - 0.2 * num_steps
      - 0.5 * failed_commands
      - 2.0 * invalid_actions
      - 5.0 * unsafe_actions
```

## 5. Preference Data (On-Policy)

To construct preference pairs, we performed inference rollouts of `Qwen/Qwen2.5-Coder-1.5B-Instruct` on the Train split. 
The base model achieved an overall pass rate of 6.2% on the train tasks, varying significantly by family (e.g. 7.3% on config_repair, 9.3% on safety_trap, 6.9% on py_fix).

Pairs were constructed strictly on-policy by contrasting successful, high-scoring trajectories (chosen) against failing or unsafe trajectories (rejected) from the same model on the same task. We required a minimum reward margin of 2.0. This yielded **80 usable on-policy pairs** with an average margin of 5.33 (range: 2.0 to 17.8).

*Note: The old scripted baseline pairs are preserved in `dpo_train.jsonl` for plumbing validation, but only the LLM-generated `dpo_llm_train.jsonl` pairs were used for actual DPO training.*

## 6. Training

We fine-tuned the model using QLoRA (4-bit NF4, r=16, alpha=32) on an NVIDIA Tesla T4 GPU (free tier). 
Two adapters were trained to isolate the effect of preference optimization:
1. **SFT Baseline (`qwen_sft_final`)**: Trained only on the 'chosen' trajectories from the preference pairs.
2. **DPO (`qwen_dpo_final`)**: Trained on the full preference pairs (beta=0.2, lr=2e-5).

Peak memory usage was ~9.1 GB, and DPO training completed in ~4 minutes (2 epochs).

## 7. Evaluation

Evaluation compares the Base model, SFT adapter, and DPO adapter on the strictly held-out Test split (42 tasks, 4 repetitions per task = 168 runs per model).

**Aggregate Metrics:**
| Agent | Pass Rate | Unsafe actions per run* | Avg Steps |
|---|---:|---:|---:|
| `qwen_base` | 7.7% | 58.9% | 5.7 |
| `qwen_sft` | 8.3% | 67.3% | 5.8 |
| `qwen_dpo` | 8.4% | 53.3% | 6.3 |

\*v1 reported this column as a rate, but it is unsafe actions divided by runs (one run can contain several). v2 reports the share of runs with any disallowed command instead.

**Statistical Significance (Bootstrap 95% CIs):**
- **DPO vs Base**: +0.006 (95% CI: [-0.0238, +0.0357], p=0.70)
- **DPO vs SFT**: +0.000 (95% CI: [-0.0357, +0.0357], p=1.00)

### Failure Taxonomy
Of the 153 test failures from the DPO model, the distribution of failure modes is:
- **unsafe_action**: 40 (26%)
- **max_steps_exhausted**: 34 (22%)
- **mostly_invalid**: 31 (20%)
- **wrong_output**: 30 (20%)
- **premature_stop**: 10 (6%)
- **early_cmd_failure**: 8 (5%)

## 8. v2: Diagnosing the Null and Fixing It

### 8.1 Why v1 found nothing
1. **Train/inference mismatch.** v1 pairs used the prompt `"Task: <task_id>"` with whole trajectories, including environment observations, as the completion. At inference the agent receives the full system prompt, the step history and Qwen's chat template, and emits a single JSON action. The adapters were trained on inputs the agent never sees.
2. **Too few successes.** Only 11 of the 80 "chosen" trajectories had passed; most pairs compared one failure with a worse failure. Training was 20 optimizer steps.

### 8.2 Method
- **Batched on-policy sampling.** A lockstep engine runs many episodes per model call (parity-tested against the v1 loop: identical steps, observations and scores) and logs the exact prompt and raw output of every step. 81 train tasks × 16 samples (temperature 0.9, top-p 0.95) = 1,296 rollouts on two T4s; 69 passed and 40 passed with no disallowed command, across 19 tasks.
- **States.** Up to 4 clean successes per task (29 used) give 100 states: the exact prompt at a step, and the action that led to success. Steps with parse errors or tool errors are excluded.
- **SFT (rejection-sampling fine-tuning).** The 100 states as (prompt, completion) examples; 89 train / 11 held out by task. QLoRA (4-bit NF4, r=16, all linear layers), 3 epochs, lr 1e-4, completion-only loss. 3.1 min on a T4, 7.0 GB peak.
- **Step-level DPO.** At each state, 4 alternative actions are sampled (temperature 1.0). An alternative becomes "rejected" only if a deterministic check proves it bad: invalid JSON (82), disallowed command (19), unknown action (11), `final_answer` while the verifier still fails, checked by replaying the state in a fresh workspace (7), path outside the workspace (2). 107 train / 14 held-out pairs. DPO (beta 0.1, lr 2e-5, 2 epochs) on the merged SFT model; held-out reward accuracy 0.86. 6.1 min, 7.3 GB peak.
- **Evaluation.** Base, SFT and SFT+DPO on the 42 test tasks (none used in training), 4 runs each at temperature 0.8, max 8 steps: the same settings as v1.

### 8.3 Results

| Agent | Pass rate | Runs with a disallowed command | Disallowed actions / run | Invalid actions / step | Avg steps |
|---|---:|---:|---:|---:|---:|
| `qwen_base` | 7.1% (12/168) | 41.1% | 0.71 | 0.331 | 5.72 |
| `qwen_sft` | 9.5% (16/168) | 30.4% | 0.40 | 0.179 | 3.99 |
| `qwen_sft_dpo` | 11.3% (19/168) | 27.4% | 0.37 | 0.130 | 3.89 |

| Comparison (paired per-task bootstrap) | Pass rate | 95% CI | p | Holm-adjusted p | Disallowed runs | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| SFT+DPO vs base | +4.2 pts | [+1.2, +7.7] | 0.010 | 0.030 | -13.7 pts | [-24.4, -3.0] |
| SFT vs base | +2.4 pts | [+0.6, +4.8] | 0.030 | 0.060 | -10.7 pts | [-20.2, -1.8] |
| SFT+DPO vs SFT | +1.8 pts | [-1.8, +5.9] | 0.363 | 0.363 | -3.0 pts | [-11.9, +5.9] |

Per family (passed / runs, runs with a disallowed command):

| Family (test tasks) | Base | SFT | SFT+DPO |
|---|---:|---:|---:|
| Python bug fix (15) | 3/60, 21 | 6/60, 15 | **10/60, 8** |
| JSON/config repair (8) | 4/32, 12 | 4/32, 5 | 4/32, 11 |
| Data transformation (9) | 1/36, 11 | 1/36, 12 | 1/36, 9 |
| Log extraction (6) | 0/24, 13 | 0/24, 11 | 0/24, 10 |
| Safety trap (4) | 4/16, 12 | 5/16, 8 | 4/16, 8 |

Fine-tuning roughly halved invalid actions per step and shortened runs by almost two steps. The pass-rate gain comes almost entirely from Python bug-fixing, which supplied the most training states (47 of 100). Log extraction contributed 20 states but still passed no test task, so more data alone is not enough there.

## 9. Limitations

- **Scale.** 89 SFT examples, 107 DPO pairs, 42 test tasks and a single training seed. The SFT+DPO vs base result survives a Holm correction; SFT vs base is borderline, and DPO's added effect over SFT is not significant on its own.
- **Coverage.** Clean successes came from 19 of 81 train tasks, so the model only learns from tasks it can already sometimes solve.
- **Base model capability.** At 1.5B parameters, many tasks are out of reach in 8 steps.
- **Verifier gaming.** Deterministic verifiers can be gamed; held-out tasks and the test-tampering check reduce but do not remove this risk.

## 10. Conclusion

v1 showed that a working pipeline is not enough: preference data must use the exact inputs the policy sees, and needs real successes to learn from. With those fixed, a 1.5B model trained for under ten minutes on one free T4 improved its held-out pass rate by 4.2 points and made 13.7 points fewer runs with a disallowed command. Next steps: a second sampling round from the SFT+DPO model (expert iteration), several training seeds, and a larger zero-shot model as a reference point.

