"""Gradio Blocks dashboard for AgentAlign Lab.

Tabs:
1. Overview — baseline vs tuned metrics, bar chart of task success rates
2. Task Explorer — browse tasks, view files and verifier config
3. Trajectory Viewer — step-by-step trajectory with verifier score
4. Preference Pair Viewer — side-by-side chosen vs rejected with score margin
5. Baseline vs Tuned — model comparison table
6. Failure Analysis — failure label distribution and example runs

Data is loaded from data/ and outputs/ directories on startup.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from agentalign.data.trajectories import load_trajectories
from agentalign.schemas import PreferencePair
from agentalign.tasks.load import load_task

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

TASKS_DIR = Path("data/tasks")
PRIMARY_SCORED_DIRS = [
    Path("data/trajectories/scored_train"),
    Path("data/trajectories/scored_val"),
    Path("data/trajectories/scored_test"),
]
LEGACY_SCORED_DIR = Path("data/trajectories/scored")
PREF_PATHS = [
    Path("data/preferences/dpo_train.jsonl"),
    Path("data/preferences/dpo_val.jsonl"),
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_tasks():
    """Load all individual task JSON files."""
    if not TASKS_DIR.exists():
        return []
    return [load_task(p) for p in sorted(TASKS_DIR.glob("*.json"))]


def _load_trajectories():
    """Load all scored trajectories from available directories."""
    trajectories = []
    scored_dirs = [p for p in PRIMARY_SCORED_DIRS if p.exists()]
    if not scored_dirs and LEGACY_SCORED_DIR.exists():
        scored_dirs = [LEGACY_SCORED_DIR]
    for scored_dir in scored_dirs:
        for path in sorted(scored_dir.glob("*.jsonl")):
            trajectories.extend(load_trajectories(path))
    return trajectories


def _load_preferences():
    """Load all preference pairs."""
    pairs = []
    for pref_path in PREF_PATHS:
        if not pref_path.exists():
            continue
        for line in pref_path.read_text().splitlines():
            if line.strip():
                pairs.append(PreferencePair.model_validate_json(line))
    return pairs


# ---------------------------------------------------------------------------
# Tab content functions
# ---------------------------------------------------------------------------

def _summarize_overview():
    """Build overview summary text."""
    tasks = _load_tasks()
    trajectories = _load_trajectories()
    preferences = _load_preferences()
    passed = sum(1 for t in trajectories if t.verifier_result and t.verifier_result.passed)
    family_counts = Counter(t.family for t in tasks)
    pass_rate = (passed / len(trajectories) * 100) if trajectories else 0.0
    lines = [
        f"Total tasks: {len(tasks)}",
        f"Task families: {dict(family_counts)}",
        f"Total trajectories: {len(trajectories)}",
        f"Total preference pairs: {len(preferences)}",
        f"Pass rate: {pass_rate:.1f}%",
    ]
    return "\n".join(lines)


def _task_table():
    return [
        [t.task_id, t.family, t.difficulty or "", t.verifier.type, len(t.files)]
        for t in _load_tasks()
    ]


def _task_ids():
    return [t.task_id for t in _load_tasks()]


def _trajectory_ids():
    return [t.run_id for t in _load_trajectories()]


def _preference_ids():
    return [p.pair_id for p in _load_preferences()]


def _show_task(task_id):
    tasks = {t.task_id: t for t in _load_tasks()}
    task = tasks.get(task_id)
    if not task:
        return "", "", {}
    files = "\n\n".join(
        f"### {f.path}\n```\n{f.content}\n```" for f in task.files
    )
    verifier = task.verifier.model_dump(mode="json")
    return task.instruction, files, verifier


def _show_trajectory(run_id):
    trajectories = {t.run_id: t for t in _load_trajectories()}
    traj = trajectories.get(run_id)
    if not traj:
        return "", {}, ""
    rows = [
        [
            step.step_index, step.thought, step.action,
            json.dumps(step.args, sort_keys=True),
            step.observation or "", step.error or "",
        ]
        for step in traj.steps
    ]
    verifier = traj.verifier_result.model_dump(mode="json") if traj.verifier_result else {}
    return rows, verifier, traj.final_answer or ""


def _show_preference(pair_id):
    pairs = {p.pair_id: p for p in _load_preferences()}
    pair = pairs.get(pair_id)
    if not pair:
        return "", "", "", ""
    scores = (
        f"chosen_score={pair.chosen_score}\n"
        f"rejected_score={pair.rejected_score}\n"
        f"margin={pair.score_margin}"
    )
    return pair.prompt, pair.chosen, pair.rejected, scores


def _model_comparison_table():
    by_agent = defaultdict(list)
    for traj in _load_trajectories():
        by_agent[traj.agent_id or traj.model or "unknown"].append(traj)
    rows = []
    for agent, trajectories in sorted(by_agent.items()):
        total = len(trajectories)
        passed = sum(1 for t in trajectories if t.verifier_result and t.verifier_result.passed)
        avg_score = (
            sum(t.verifier_result.score for t in trajectories if t.verifier_result) / total
            if total else 0.0
        )
        avg_steps = sum(len(t.steps) for t in trajectories) / total if total else 0.0
        rows.append([
            agent, total,
            f"{passed / total * 100:.1f}%",
            f"{avg_score:.2f}",
            f"{avg_steps:.1f}",
        ])
    return rows


def _failure_table():
    counter = Counter()
    examples = {}
    for traj in _load_trajectories():
        if not traj.verifier_result or traj.verifier_result.passed:
            continue
        tags = traj.verifier_result.failure_tags or ["unknown_failure"]
        for tag in tags:
            counter[tag] += 1
            examples.setdefault(tag, traj.run_id)
    return [[tag, count, examples[tag]] for tag, count in counter.most_common()]


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app():
    """Build and return the Gradio Blocks dashboard."""
    import gradio as gr

    with gr.Blocks(title="AgentAlign Dashboard") as app:
        gr.Markdown("# 🧠 AgentAlign Lab Dashboard")

        # Tab 1: Overview
        with gr.Tab("Overview"):
            refresh_btn = gr.Button("🔄 Refresh")
            summary = gr.Textbox(label="Metrics Summary", lines=8)
            refresh_btn.click(_summarize_overview, outputs=summary)
            app.load(_summarize_overview, outputs=summary)

        # Tab 2: Task Explorer
        with gr.Tab("Task Explorer"):
            gr.Dataframe(
                headers=["task_id", "family", "difficulty", "verifier", "files"],
                value=_task_table(),
                interactive=False,
            )
            task_selector = gr.Dropdown(label="Task", choices=_task_ids())
            instruction_box = gr.Textbox(label="Instruction", lines=3)
            files_md = gr.Markdown()
            verifier_json = gr.JSON(label="Verifier")
            task_selector.change(
                _show_task,
                inputs=task_selector,
                outputs=[instruction_box, files_md, verifier_json],
            )

        # Tab 3: Trajectory Viewer
        with gr.Tab("Trajectory Viewer"):
            traj_selector = gr.Dropdown(label="Trajectory", choices=_trajectory_ids())
            steps_df = gr.Dataframe(
                headers=["step", "thought", "action", "args", "observation", "error"],
                interactive=False,
            )
            vr_json = gr.JSON(label="Verifier Result")
            final_ans = gr.Textbox(label="Final Answer", lines=2)
            traj_selector.change(
                _show_trajectory,
                inputs=traj_selector,
                outputs=[steps_df, vr_json, final_ans],
            )

        # Tab 4: Preference Pair Viewer
        with gr.Tab("Preference Pair Viewer"):
            pref_selector = gr.Dropdown(label="Preference Pair", choices=_preference_ids())
            prompt_box = gr.Textbox(label="Prompt", lines=8)
            chosen_box = gr.Textbox(label="Chosen", lines=8)
            rejected_box = gr.Textbox(label="Rejected", lines=8)
            scores_box = gr.Textbox(label="Scores", lines=3)
            pref_selector.change(
                _show_preference,
                inputs=pref_selector,
                outputs=[prompt_box, chosen_box, rejected_box, scores_box],
            )

        # Tab 5: Baseline vs Tuned
        with gr.Tab("Baseline vs Tuned"):
            gr.Dataframe(
                headers=["agent", "runs", "pass_rate", "avg_score", "avg_steps"],
                value=_model_comparison_table(),
                interactive=False,
            )

        # Tab 6: Failure Analysis
        with gr.Tab("Failure Analysis"):
            gr.Dataframe(
                headers=["failure_tag", "count", "example_run"],
                value=_failure_table(),
                interactive=False,
            )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
