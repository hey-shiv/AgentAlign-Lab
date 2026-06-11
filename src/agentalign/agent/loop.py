import datetime
import time
import uuid
from typing import Callable

from agentalign.agent.parser import parse_action
from agentalign.agent.prompts import build_system_prompt, format_history
from agentalign.agent.tools import ToolExecutor
from agentalign.schemas import Step, Task, Trajectory
from agentalign.tasks.workspace import TaskWorkspace


def run_agent_loop(
    task: Task,
    model_callable: Callable[[str], str],
    agent_id: str = "baseline",
    max_steps: int = 8,
    model_name: str = "unknown"
) -> Trajectory:

    run_id = f"run_{uuid.uuid4().hex[:8]}_{task.task_id}_{agent_id}"
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    trajectory = Trajectory(
        run_id=run_id,
        task_id=task.task_id,
        agent_id=agent_id,
        model=model_name,
        started_at=started_at,
        metadata={"max_steps": max_steps}
    )

    system_prompt = build_system_prompt(task)
    history = []
    final_answer = None

    with TaskWorkspace(task) as workspace_path:
        executor = ToolExecutor(workspace_path, forbidden_commands=task.forbidden_commands)

        for step_idx in range(1, max_steps + 1):
            prompt = system_prompt + "\n\n" + format_history(history) + "\n\nAction:"

            step_start = time.monotonic()
            raw_text = model_callable(prompt)
            latency_ms = int((time.monotonic() - step_start) * 1000)

            action_result, error_msg = parse_action(raw_text)

            if error_msg:
                # Invalid action
                history.append({
                    "text": raw_text,
                    "error": error_msg
                })
                trajectory.steps.append(Step(
                    step_index=step_idx,
                    thought="<failed to parse>",
                    action="<invalid>",
                    error=error_msg,
                    latency_ms=latency_ms
                ))
                continue

            action = action_result

            # Action is valid, execute
            observation = executor.execute(action.name, action.args)

            history.append({
                "text": raw_text,
                "observation": observation
            })

            trajectory.steps.append(Step(
                step_index=step_idx,
                thought=raw_text.split('"thought":')[1].split('",')[0].strip(' "') if '"thought":' in raw_text else "extracted", # naive extraction for thought
                action=action.name,
                args=action.args,
                observation=observation,
                latency_ms=latency_ms
            ))

            if action.name == "final_answer":
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
