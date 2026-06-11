import json
from collections import Counter, defaultdict
from pathlib import Path

import gradio as gr

from agentalign.data.trajectories import load_trajectories
from agentalign.schemas import PreferencePair
from agentalign.tasks.load import load_task

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


def load_tasks():
    return [load_task(path) for path in sorted(TASKS_DIR.glob("*.json"))]


def load_dashboard_data():
    trajectories = []
    scored_dirs = [path for path in PRIMARY_SCORED_DIRS if path.exists()]
    if not scored_dirs and LEGACY_SCORED_DIR.exists():
        scored_dirs = [LEGACY_SCORED_DIR]
    for scored_dir in scored_dirs:
        for path in sorted(scored_dir.glob("*.jsonl")):
            trajectories.extend(load_trajectories(path))
    return trajectories


def load_preferences():
    pairs = []
    for pref_path in PREF_PATHS:
        if not pref_path.exists():
            continue
        for line in pref_path.read_text().splitlines():
            if line.strip():
                pairs.append(PreferencePair.model_validate_json(line))
    return pairs


def summarize_overview():
    tasks = load_tasks()
    trajectories = load_dashboard_data()
    preferences = load_preferences()
    passed = sum(1 for traj in trajectories if traj.verifier_result and traj.verifier_result.passed)
    unsafe = sum(
        1
        for traj in trajectories
        for step in traj.steps
        if step.error == "forbidden_command" or "forbidden" in (step.observation or "").lower()
    )
    family_counts = Counter(task.family for task in tasks)
    pass_rate = (passed / len(trajectories) * 100) if trajectories else 0.0
    lines = [
        f"Total tasks: {len(tasks)}",
        f"Task families: {dict(family_counts)}",
        f"Total trajectories: {len(trajectories)}",
        f"Total preference pairs: {len(preferences)}",
        f"Pass rate: {pass_rate:.1f}%",
        f"Unsafe action signals: {unsafe}",
    ]
    return "\n".join(lines)


def task_table():
    return [
        [task.task_id, task.family, task.difficulty or "", task.verifier.type, len(task.files)]
        for task in load_tasks()
    ]


def task_ids():
    return [task.task_id for task in load_tasks()]


def trajectory_ids():
    return [traj.run_id for traj in load_dashboard_data()]


def preference_ids():
    return [pair.pair_id for pair in load_preferences()]


def show_task(task_id):
    tasks = {task.task_id: task for task in load_tasks()}
    task = tasks.get(task_id)
    if not task:
        return "", "", {}
    files = "\n\n".join(f"### {file.path}\n```\n{file.content}\n```" for file in task.files)
    verifier = task.verifier.model_dump(mode="json")
    return task.instruction, files, verifier


def show_trajectory(run_id):
    trajectories = {traj.run_id: traj for traj in load_dashboard_data()}
    traj = trajectories.get(run_id)
    if not traj:
        return "", {}, ""
    rows = [
        [
            step.step_index,
            step.thought,
            step.action,
            json.dumps(step.args, sort_keys=True),
            step.observation or "",
            step.error or "",
        ]
        for step in traj.steps
    ]
    verifier = traj.verifier_result.model_dump(mode="json") if traj.verifier_result else {}
    return rows, verifier, traj.final_answer or ""


def show_preference(pair_id):
    pairs = {pair.pair_id: pair for pair in load_preferences()}
    pair = pairs.get(pair_id)
    if not pair:
        return "", "", "", ""
    scores = (
        f"chosen_score={pair.chosen_score}\n"
        f"rejected_score={pair.rejected_score}\n"
        f"margin={pair.score_margin}"
    )
    return pair.prompt, pair.chosen, pair.rejected, scores


def model_comparison_table():
    by_agent = defaultdict(list)
    for traj in load_dashboard_data():
        by_agent[traj.agent_id or traj.model or "unknown"].append(traj)

    rows = []
    for agent, trajectories in sorted(by_agent.items()):
        total = len(trajectories)
        passed = sum(1 for traj in trajectories if traj.verifier_result and traj.verifier_result.passed)
        avg_score = (
            sum(traj.verifier_result.score for traj in trajectories if traj.verifier_result) / total
            if total
            else 0.0
        )
        avg_steps = sum(len(traj.steps) for traj in trajectories) / total if total else 0.0
        rows.append([agent, total, f"{passed / total * 100:.1f}%", f"{avg_score:.2f}", f"{avg_steps:.1f}"])
    return rows


def failure_table():
    counter = Counter()
    examples = {}
    for traj in load_dashboard_data():
        if not traj.verifier_result or traj.verifier_result.passed:
            continue
        tags = traj.verifier_result.failure_tags or ["unknown_failure"]
        for tag in tags:
            counter[tag] += 1
            examples.setdefault(tag, traj.run_id)
    return [[tag, count, examples[tag]] for tag, count in counter.most_common()]


def build_app():
    with gr.Blocks(title="AgentAlign Dashboard") as app:
        gr.Markdown("# AgentAlign Lab Dashboard")

        with gr.Tab("Overview"):
            refresh = gr.Button("Refresh")
            summary = gr.Textbox(label="Metrics Summary", lines=8)
            refresh.click(summarize_overview, outputs=summary)
            app.load(summarize_overview, outputs=summary)

        with gr.Tab("Task Explorer"):
            gr.Dataframe(
                headers=["task_id", "family", "difficulty", "verifier", "files"],
                value=task_table(),
                interactive=False,
            )
            task_selector = gr.Dropdown(label="Task", choices=task_ids())
            instruction = gr.Textbox(label="Instruction", lines=3)
            files = gr.Markdown()
            verifier = gr.JSON(label="Verifier")
            task_selector.change(show_task, inputs=task_selector, outputs=[instruction, files, verifier])

        with gr.Tab("Trajectory Viewer"):
            trajectory_selector = gr.Dropdown(label="Trajectory", choices=trajectory_ids())
            steps = gr.Dataframe(
                headers=["step", "thought", "action", "args", "observation", "error"],
                interactive=False,
            )
            verifier_result = gr.JSON(label="Verifier Result")
            final_answer = gr.Textbox(label="Final Answer", lines=2)
            trajectory_selector.change(
                show_trajectory,
                inputs=trajectory_selector,
                outputs=[steps, verifier_result, final_answer],
            )

        with gr.Tab("Preference Pair Viewer"):
            pref_selector = gr.Dropdown(label="Preference Pair", choices=preference_ids())
            prompt = gr.Textbox(label="Prompt", lines=8)
            chosen = gr.Textbox(label="Chosen", lines=8)
            rejected = gr.Textbox(label="Rejected", lines=8)
            scores = gr.Textbox(label="Scores", lines=3)
            pref_selector.change(show_preference, inputs=pref_selector, outputs=[prompt, chosen, rejected, scores])

        with gr.Tab("Baseline vs Tuned"):
            gr.Dataframe(
                headers=["agent", "runs", "pass_rate", "avg_score", "avg_steps"],
                value=model_comparison_table(),
                interactive=False,
            )

        with gr.Tab("Failure Analysis"):
            gr.Dataframe(
                headers=["failure_tag", "count", "example_run"],
                value=failure_table(),
                interactive=False,
            )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
