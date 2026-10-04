"""Training data for v2 from step-level rollout logs.

SFT (rejection-sampling fine-tuning): every clean step of a successful trajectory, as
(exact prompt, raw completion).

DPO: at a state taken from a successful trajectory, the action that led to success is
"chosen"; an alternative action sampled at the same state is "rejected" only when a
deterministic check proves it bad (invalid JSON, disallowed command, editing tests, an
unsafe path, or calling final_answer while the verifier still fails).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from agentalign.schemas import Task, Trajectory
from agentalign.v2.engine import premature_final_answer, same_action, static_reject_reason


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def load_trajectories(paths: Iterable[str | Path]) -> list[Trajectory]:
    out: list[Trajectory] = []
    for p in paths:
        out.extend(Trajectory.model_validate(r) for r in read_jsonl(p))
    return out


def is_clean_success(t: Trajectory) -> bool:
    vr = t.verifier_result
    return bool(vr and vr.passed and vr.unsafe_actions == 0)


def good_steps(t: Trajectory) -> list[int]:
    """Indices of steps in a successful trajectory worth imitating."""
    keep = []
    for i, step in enumerate(t.steps):
        if step.error or step.action == "<invalid>":
            continue
        obs = (step.observation or "").strip()
        if obs.startswith("Error"):
            continue
        keep.append(i)
    return keep


def select_successes(trajs: list[Trajectory], max_per_task: int, seed: int = 0) -> list[Trajectory]:
    """At most `max_per_task` clean successes per task, so easy tasks don't dominate."""
    rng = random.Random(seed)
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajs:
        if is_clean_success(t) and t.metadata.get("step_io"):
            by_task[t.task_id].append(t)
    chosen: list[Trajectory] = []
    for task_id in sorted(by_task):
        group = sorted(by_task[task_id], key=lambda t: (len(t.steps), t.run_id))
        rng.shuffle(group)
        chosen.extend(sorted(group[:max_per_task], key=lambda t: t.run_id))
    return chosen


def build_states(successes: list[Trajectory]) -> list[dict]:
    """One state per good step: the prompt and the action that led to success."""
    states = []
    for t in successes:
        io = {row["step_index"]: row for row in t.metadata["step_io"]}
        for i in good_steps(t):
            step = t.steps[i]
            row = io.get(step.step_index)
            if row is None:
                continue
            states.append({
                "state_id": f"{t.run_id}:{step.step_index}",
                "task_id": t.task_id,
                "run_id": t.run_id,
                "step_index": step.step_index,
                "prompt": row["prompt"],
                "chosen": row["completion"],
                "prefix_steps": [s.model_dump() for s in t.steps[:i]],
            })
    return states


def build_sft_examples(states: list[dict]) -> list[dict]:
    seen = set()
    rows = []
    for s in states:
        key = (s["prompt"], s["chosen"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({"task_id": s["task_id"], "prompt": s["prompt"], "completion": s["chosen"]})
    return rows


def label_alternative(task: Task, state: dict, alternative: str) -> str | None:
    """Why `alternative` is verifiably worse than the chosen action, else None."""
    if same_action(alternative, state["chosen"]):
        return None
    reason = static_reject_reason(alternative, task)
    if reason:
        return reason
    from agentalign.agent.parser import parse_action
    from agentalign.schemas import Step

    action, _ = parse_action(alternative)
    if action is not None and action.action == "final_answer":
        prefix = [Step.model_validate(s) for s in state["prefix_steps"]]
        if premature_final_answer(task, prefix):
            return "premature_final_answer"
    return None


def build_dpo_pairs(
    states: list[dict],
    alternatives: dict[str, list[str]],
    tasks: dict[str, Task],
    max_pairs_per_state: int = 2,
) -> list[dict]:
    pairs = []
    for s in states:
        task = tasks[s["task_id"]]
        used = 0
        seen_rejected = set()
        for alt in alternatives.get(s["state_id"], []):
            if used >= max_pairs_per_state:
                break
            key = alt.strip()
            if key in seen_rejected:
                continue
            reason = label_alternative(task, s, alt)
            if reason is None:
                continue
            seen_rejected.add(key)
            used += 1
            pairs.append({
                "pair_id": f"{s['state_id']}:{used}",
                "task_id": s["task_id"],
                "step_index": s["step_index"],
                "prompt": s["prompt"],
                "chosen": s["chosen"],
                "rejected": alt,
                "reject_reason": reason,
                "source": "step_verifier",
            })
    return pairs


def split_by_task(rows: list[dict], eval_fraction: float, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Hold out whole tasks so the eval loss measures generalisation to unseen tasks."""
    tasks = sorted({r["task_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(tasks)
    n_eval = int(round(len(tasks) * eval_fraction))
    if len(tasks) >= 4:
        n_eval = max(1, n_eval)
    held = set(tasks[:n_eval])
    return [r for r in rows if r["task_id"] not in held], [r for r in rows if r["task_id"] in held]
