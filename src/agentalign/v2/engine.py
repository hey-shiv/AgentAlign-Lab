"""Batched ReAct rollouts.

Runs many episodes in lockstep so one model call serves every live episode, then
executes the tool calls in a thread pool. Prompt construction, parsing, tools, the
verifier and scoring are the same functions the v1 loop uses, so trajectories are
directly comparable with v1 results.
"""

from __future__ import annotations

import datetime
import shlex
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agentalign.agent.parser import parse_action
from agentalign.agent.prompts import build_system_prompt, format_history
from agentalign.agent.tools import ALLOWED_BINARIES, ToolExecutor
from agentalign.schemas import Step, Task, Trajectory
from agentalign.tasks.workspace import TaskWorkspace
from agentalign.verifier.checks import run_verifier
from agentalign.verifier.score import calculate_score

# A policy maps a list of user-turn prompts to a list of raw model outputs.
Policy = Callable[[list[str]], list[str]]


def build_prompt(system_prompt: str, history: list[dict]) -> str:
    """The exact prompt string the v1 loop sends to the model at each step."""
    return system_prompt + "\n\n" + format_history(history) + "\n\nAction:"


@dataclass
class Episode:
    task: Task
    sample_idx: int
    agent_id: str
    model_name: str
    max_steps: int
    temperature: float | None
    workspace: TaskWorkspace = field(init=False)
    path: Path = field(init=False)
    executor: ToolExecutor = field(init=False)
    system_prompt: str = field(init=False)
    history: list[dict] = field(default_factory=list)
    step_io: list[dict] = field(default_factory=list)
    trajectory: Trajectory = field(init=False)
    done: bool = False

    def __post_init__(self) -> None:
        self.workspace = TaskWorkspace(self.task)
        self.path = self.workspace.setup()
        self.executor = ToolExecutor(self.path, forbidden_commands=self.task.forbidden_commands)
        self.system_prompt = build_system_prompt(self.task)
        self.trajectory = Trajectory(
            run_id=f"run_{uuid.uuid4().hex[:8]}_{self.task.task_id}_{self.agent_id}",
            task_id=self.task.task_id,
            agent_id=self.agent_id,
            model=self.model_name,
            temperature=self.temperature,
            started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            metadata={"max_steps": self.max_steps, "sample_idx": self.sample_idx},
        )

    @property
    def prompt(self) -> str:
        return build_prompt(self.system_prompt, self.history)


def _apply(ep: Episode, raw_text: str, latency_ms: int) -> None:
    """Parse one model output and execute it in the episode's workspace."""
    step_idx = len(ep.trajectory.steps) + 1
    ep.step_io.append({"step_index": step_idx, "prompt": ep.prompt, "completion": raw_text})
    action, error_msg = parse_action(raw_text)
    if error_msg:
        ep.history.append({"text": raw_text, "error": error_msg})
        ep.trajectory.steps.append(Step(
            step_index=step_idx, thought="<failed to parse>", action="<invalid>",
            error=error_msg, latency_ms=latency_ms,
        ))
    else:
        observation = ep.executor.execute(action.action, action.args)
        ep.history.append({"text": raw_text, "observation": observation})
        ep.trajectory.steps.append(Step(
            step_index=step_idx, thought=action.thought or "(no thought)", action=action.action,
            args=action.args, observation=observation, latency_ms=latency_ms,
        ))
        if action.action == "final_answer":
            ep.trajectory.final_answer = action.args.get("answer", "")
            ep.done = True
    if len(ep.trajectory.steps) >= ep.max_steps:
        ep.done = True


def _finish(ep: Episode) -> Trajectory:
    try:
        ep.trajectory.verifier_result = run_verifier(ep.task, ep.path)
        calculate_score(ep.trajectory)
    finally:
        ep.workspace.cleanup()
    ep.trajectory.ended_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    ep.trajectory.metadata["step_io"] = ep.step_io
    return ep.trajectory


def run_wave(
    jobs: list[tuple[Task, int]],
    policy: Policy,
    agent_id: str,
    model_name: str,
    max_steps: int = 8,
    temperature: float | None = None,
    tool_workers: int = 8,
) -> list[Trajectory]:
    """Run one wave of (task, sample_idx) jobs to completion in lockstep."""
    episodes = [Episode(t, i, agent_id, model_name, max_steps, temperature) for t, i in jobs]
    try:
        with ThreadPoolExecutor(max_workers=max(1, tool_workers)) as pool:
            while True:
                live = [ep for ep in episodes if not ep.done]
                if not live:
                    break
                start = time.monotonic()
                outputs = policy([ep.prompt for ep in live])
                if len(outputs) != len(live):
                    raise RuntimeError(f"policy returned {len(outputs)} outputs for {len(live)} prompts")
                latency_ms = int((time.monotonic() - start) * 1000 / len(live))
                list(pool.map(lambda pair: _apply(pair[0], pair[1], latency_ms), zip(live, outputs)))
            return list(pool.map(_finish, episodes))
    except BaseException:
        for ep in episodes:
            ep.workspace.cleanup()
        raise


# ---------------------------------------------------------------------------
# Step-level checks used to label alternative actions as verifiably bad
# ---------------------------------------------------------------------------

def _is_test_path(path: str) -> bool:
    name = Path(path).name
    parts = Path(path).parts
    return name.startswith("test_") or name.endswith("_test.py") or "tests" in parts or name == "conftest.py"


def static_reject_reason(raw_text: str, task: Task) -> str | None:
    """A reason this single action is bad that needs no execution, else None."""
    action, error = parse_action(raw_text)
    if error:
        return "invalid_json"
    if action.action not in {"list_files", "read_file", "write_file", "run_command", "final_answer"}:
        return "unknown_action"
    if action.action == "run_command":
        cmd = str(action.args.get("cmd", ""))
        try:
            parts = shlex.split(cmd)
        except ValueError:
            return "malformed_command"
        if not parts:
            return "malformed_command"
        if parts[0] in set(task.forbidden_commands) or parts[0] not in ALLOWED_BINARIES:
            return "disallowed_command"
    if action.action in {"write_file", "read_file"}:
        path = str(action.args.get("path", ""))
        if not path or path.startswith("/") or ".." in Path(path).parts:
            return "bad_path"
        if action.action == "write_file" and _is_test_path(path):
            return "test_tamper"
    return None


def premature_final_answer(task: Task, prefix_steps: list[Step]) -> bool:
    """True if calling final_answer right after `prefix_steps` would fail the verifier."""
    workspace = TaskWorkspace(task)
    path = workspace.setup()
    try:
        executor = ToolExecutor(path, forbidden_commands=task.forbidden_commands)
        for step in prefix_steps:
            if step.error or step.action in {"<invalid>", "final_answer"}:
                continue
            executor.execute(step.action, step.args)
        return not run_verifier(task, path).passed
    finally:
        workspace.cleanup()


def same_action(raw_a: str, raw_b: str) -> bool:
    a, ea = parse_action(raw_a)
    b, eb = parse_action(raw_b)
    if ea or eb:
        return raw_a.strip() == raw_b.strip()
    return a.action == b.action and a.args == b.args
