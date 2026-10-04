# AgentAlign Lab: Technical Report

**Title:** Verifier-Guided Preference Data for Reliable Terminal Agents  
**Author:** AgentAlign Team

## Abstract

Terminal agents often look confident while failing tests, producing invalid tool calls, or taking unsafe actions. AgentAlign Lab implements a compact local feedback loop: deterministic terminal tasks, structured agent traces, verifier scoring, high-margin preference-pair generation, split-aware evaluation, and a Gradio dashboard. In this iteration, we replace scripted baselines with on-policy preference pairs generated from `Qwen/Qwen2.5-Coder-1.5B-Instruct` rollouts, and evaluate QLoRA DPO against both the base model and an SFT baseline.

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
| Agent | Pass Rate | Unsafe Rate | Avg Steps |
|---|---:|---:|---:|
| `qwen_base` | 7.7% | 58.9% | 5.7 |
| `qwen_sft` | 8.3% | 67.3% | 5.8 |
| `qwen_dpo` | 8.4% | 53.3% | 6.3 |

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

## 8. Limitations

- **Compute/Scale**: The dataset size (80 pairs) is a proof-of-concept. Real-world terminal agents require tens of thousands of diverse pairs.
- **Base Model Capability**: At 1.5B parameters, Qwen2.5-Coder struggles fundamentally with certain agentic reasoning leaps, meaning many tasks had no successful rollouts to form pairs from.
- **Verifier Gaming**: Deterministic verifiers can be gamed, though held-out test tasks reduce this risk.

## 9. Conclusion

AgentAlign Lab implements the core MVP loop: task suite, local agent harness, deterministic verifiers, on-policy preference pair generation, split-aware evaluation, and a Gradio dashboard. 

The QLoRA DPO adapter achieved an 8.4% pass rate compared to the SFT baseline (8.3%) and the Base model (7.7%). Because the 95% confidence intervals span zero, we conclude there is no statistically significant improvement on this limited scale. However, the pipeline successfully executed end-to-end, producing a rigorous, reproducible, and verifiable result that validates the AgentAlign Lab data-generation and evaluation machinery.
