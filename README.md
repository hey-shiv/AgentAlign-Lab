<p align="left">
  <small><code>RESEARCH &middot; PREFERENCE LEARNING &middot; AGENTIC SYSTEMS</code></small>
</p>

<h1>Agent<em>Align</em><br>Lab</h1>

<p>
  A pipeline for verifier-guided preference learning in terminal agents.<br>
  No human raters. Structured feedback at scale.
</p>

<hr>

#### <small><code>MOTIVATION</code></small>

Current RLHF pipelines depend on human preference labels — expensive, inconsistent, and hard to scale for agentic settings. AgentAlign Lab replaces human raters with deterministic verifiers wherever the task has an unambiguous correct answer: code execution, arithmetic, structured retrieval. The feedback loop closes automatically.

<hr>

#### <small><code>PIPELINE</code></small>

<table>
  <tr>
    <td valign="top"><code>01</code></td>
    <td valign="top"><strong>Task Generation</strong><br>Synthetic task datasets across verifiable domains — code, math, structured data retrieval.</td>
  </tr>
  <tr>
    <td valign="top"><code>02</code></td>
    <td valign="top"><strong>ReAct Agent Loop</strong><br>Agent observes, reasons, and acts across tool calls. Full trajectory is recorded.</td>
  </tr>
  <tr>
    <td valign="top"><code>03</code></td>
    <td valign="top"><strong>Deterministic Verifier</strong><br>Pure functions. Input: trajectory. Output: scalar score. No learned reward model.</td>
  </tr>
  <tr>
    <td valign="top"><code>04</code></td>
    <td valign="top"><strong>DPO Preference Pairs</strong><br>Scored trajectories are paired — chosen vs rejected — for direct preference optimization.</td>
  </tr>
  <tr>
    <td valign="top"><code>05</code></td>
    <td valign="top"><strong>QLoRA Fine-Tuning</strong><br>TRL-based training on Apple Silicon or free-tier GPU. No dedicated compute required.</td>
  </tr>
  <tr>
    <td valign="top"><code>06</code></td>
    <td valign="top"><strong>Evaluation + Dashboard</strong><br>Evaluation harness and Gradio dashboard for inspecting trajectories and training outcomes.</td>
  </tr>
</table>

<hr>

#### <small><code>DESIGN DECISIONS</code></small>

<table>
  <tr>
    <td valign="top" width="50%">
      <small><code>WHY DPO, NOT PPO?</code></small><br><br>
      <strong>Stability.</strong> DPO avoids the instability and hyperparameter sensitivity of online RL — a natural fit for synthetic pair construction.
    </td>
    <td valign="top" width="50%">
      <small><code>WHY DETERMINISTIC VERIFIERS?</code></small><br><br>
      <strong>Interpretability.</strong> Rule-based scorers are more reliable and cheaper than a learned reward model for tasks with ground truth.
    </td>
  </tr>
  <tr>
    <td valign="top" width="50%">
      <small><code>WHY QLORA?</code></small><br><br>
      <strong>Accessibility.</strong> Runs on M-series chips and Colab free tier. The entire loop — generation to fine-tune — needs no paid GPU.
    </td>
    <td valign="top" width="50%">
      <small><code>WHY MODULAR STAGES?</code></small><br><br>
      <strong>Inspectability.</strong> Stages write to disk in defined schemas. Swap any component; resume interrupted runs cleanly.
    </td>
  </tr>
</table>

<hr>

#### <small><code>STRUCTURE</code></small>

```text
AgentAlign-Lab/
  tasks/       - task generators and dataset schemas
  agent/       - ReAct loop implementation
  verifiers/   - deterministic verifier suite
  preference/  - DPO pair construction
  training/    - QLoRA fine-tuning via TRL
  eval/        - evaluation harness
  dashboard/   - Gradio interface
  configs/     - experiment configs (YAML)
  scripts/     - end-to-end run scripts
```

<hr>

#### <small><code>STATUS</code></small>

<p>
  <code>task generation - done</code>&nbsp;
  <code>react loop - done</code>&nbsp;
  <code>verifiers - done</code>&nbsp;
  <code>dpo pairs - done</code><br><br>
  <code>fine-tuning - done</code>&nbsp;
  <code>eval - done</code>
</p>

<hr>

#### <small><code>RESEARCH QUESTION</code></small>

> *To what extent can verifier-guided feedback — without any human annotation — produce agents that generalise across task types rather than overfit to the verifier's scoring function?*

Targeting a workshop paper or technical report as first output.

<hr>

<p>
  Shiv &middot; VIT-AP University, B.Tech CS (AI/ML) &middot; Batch 2027<br>
  <a href="https://hey-shiv.github.io">hey-shiv.github.io</a> &middot; <a href="https://x.com/NaadhLabs">@NaadhLabs</a>
</p>
