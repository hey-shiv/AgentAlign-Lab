"""ReAct agent loop.

Runs a task through the full agent cycle:
1. Create workspace
2. For each step: build prompt → generate → parse → validate → execute
3. Run verifier
4. Return Trajectory
"""

import datetime
import time
import uuid
from typing import Callable

from agentalign.agent.parser import parse_action
from agentalign.agent.prompts import build_system_prompt, format_history
from agentalign.agent.tools import ToolExecutor, tool_list_files
from agentalign.schemas import Step, Task, Trajectory
from agentalign.tasks.workspace import TaskWorkspace


def run_agent_loop(
    task: Task,
    model_callable: Callable[[str], str],
    agent_id: str = "baseline",
    max_steps: int = 8,
    model_name: str = "unknown",
) -> Trajectory:
    """Execute the full ReAct loop on a single task.

    Args:
        task: The task to solve.
        model_callable: A function that takes a prompt string and returns
            the model's raw text output.
        agent_id: Identifier for the agent variant (e.g. 'baseline', 'bad_model').
        max_steps: Maximum number of agent steps before forced stop.
        model_name: Human-readable model name stored in the trajectory.

    Returns:
        A fully populated Trajectory including verifier results.
    """
    run_id = f"run_{uuid.uuid4().hex[:8]}_{task.task_id}_{agent_id}"
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    trajectory = Trajectory(
        run_id=run_id,
        task_id=task.task_id,
        agent_id=agent_id,
        model=model_name,
        started_at=started_at,
        metadata={"max_steps": max_steps},
    )

    system_prompt = build_system_prompt(task)
    history: list[dict] = []
    final_answer = None

    with TaskWorkspace(task) as workspace_path:
        executor = ToolExecutor(workspace_path, forbidden_commands=task.forbidden_commands)

        for step_idx in range(1, max_steps + 1):
            # Build the full prompt for this turn
            prompt = system_prompt + "\n\n" + format_history(history) + "\n\nAction:"

            # Call the model
            step_start = time.monotonic()
            raw_text = model_callable(prompt)
            latency_ms = int((time.monotonic() - step_start) * 1000)

            # Parse the action
            action_result, error_msg = parse_action(raw_text)

            if error_msg:
                # Invalid action — log it and continue
                history.append({"text": raw_text, "error": error_msg})
                trajectory.steps.append(Step(
                    step_index=step_idx,
                    thought="<failed to parse>",
                    action="<invalid>",
                    error=error_msg,
                    latency_ms=latency_ms,
                ))
                continue

            action = action_result

            # Execute the tool
            observation = executor.execute(action.action, action.args)

            history.append({"text": raw_text, "observation": observation})

            # Extract thought from the parsed action
            thought = action.thought if action.thought else "(no thought)"

            trajectory.steps.append(Step(
                step_index=step_idx,
                thought=thought,
                action=action.action,
                args=action.args,
                observation=observation,
                latency_ms=latency_ms,
            ))

            if action.action == "final_answer":
                final_answer = action.args.get("answer", "")
                break

        # Run verifier
        from agentalign.verifier.checks import run_verifier
        from agentalign.verifier.score import calculate_score

        verifier_result = run_verifier(task, workspace_path)
        trajectory.verifier_result = verifier_result
        trajectory.final_answer = final_answer

        # Calculate composite score
        calculate_score(trajectory)

    trajectory.ended_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return trajectory
