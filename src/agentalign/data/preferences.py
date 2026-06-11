import json
import uuid
from collections import defaultdict
from pathlib import Path

from agentalign.agent.prompts import build_system_prompt
from agentalign.schemas import Preference, PreferencePair, Trajectory


def build_prompt_from_trajectory(traj: Trajectory, task) -> str:
    # MVP DPO format: prompt, chosen, rejected.
    # For training, prompt includes task instructions + initial state.
    # The generation is the full trace of actions.
    prompt = build_system_prompt(task) + "\n\nAction:"
    return prompt

def format_trajectory_actions(traj: Trajectory) -> str:
    # Concatenate the actions and observations.
    # In DPO, we want to train the model to output the whole sequence or just the next step.
    # The MVP dictates trajectory-level preferences: full action traces.
    res = []
    for step in traj.steps:
        # Reconstruct the raw JSON the model would have output
        if step.error:
            # For rejected traces, it might include invalid actions
            res.append(f'{{\n  "thought": "{step.thought}",\n  "action": "{step.action}",\n  "args": {json.dumps(step.args)}\n}}')
        else:
            res.append(f'{{\n  "thought": "{step.thought}",\n  "action": "{step.action}",\n  "args": {json.dumps(step.args)}\n}}')
            res.append(f"Observation:\n{step.observation}")
    return "\n\n".join(res)

def generate_preference_pairs(
    trajectories: list[Trajectory],
    tasks_dict: dict,
    min_margin: float = 2.0
) -> list[PreferencePair]:

    # Group by task_id
    grouped = defaultdict(list)
    for t in trajectories:
        if t.verifier_result:
            grouped[t.task_id].append(t)

    pairs = []

    for task_id, trajs in grouped.items():
        if task_id not in tasks_dict:
            continue
        task = tasks_dict[task_id]
        prompt = build_prompt_from_trajectory(trajs[0], task)

        # Sort trajectories by score descending
        trajs.sort(key=lambda x: x.verifier_result.score, reverse=True)

        # Compare highest scoring with lowest scoring
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                chosen_traj = trajs[i]
                rejected_traj = trajs[j]

                margin = chosen_traj.verifier_result.score - rejected_traj.verifier_result.score
                if margin >= min_margin:
                    # Valid pair
                    pair = PreferencePair(
                        pair_id=f"pref_{uuid.uuid4().hex[:8]}",
                        task_id=task_id,
                        prompt=prompt,
                        chosen=format_trajectory_actions(chosen_traj),
                        rejected=format_trajectory_actions(rejected_traj),
                        chosen_score=chosen_traj.verifier_result.score,
                        rejected_score=rejected_traj.verifier_result.score,
                        score_margin=margin
                    )
                    pairs.append(pair)
                    # For MVP, maybe we just take 1-2 pairs per task to avoid imbalance
                    break

    return pairs

def save_preferences(pairs: list[PreferencePair], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for p in pairs:
            f.write(p.model_dump_json() + '\n')


def build_preferences_from_trajectories(
    good_trajectories: list[Trajectory], bad_trajectories: list[Trajectory]
) -> list[Preference]:
    preferences = []
    for good, bad in zip(good_trajectories, bad_trajectories):
        if good.task_id != bad.task_id:
            continue
        preferences.append(
            Preference(
                prompt=f"Task: {good.task_id}",
                chosen=format_trajectory_actions(good) or good.final_answer or "",
                rejected=format_trajectory_actions(bad) or bad.final_answer or "",
            )
        )
    return preferences
