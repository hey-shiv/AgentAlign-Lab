import subprocess
import time

from agentalign.schemas import Task, VerifierResult


def run_verifier(task: Task, workspace) -> VerifierResult:
    start = time.time()

    result = subprocess.run(
        task.verifier.command.split(),
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=task.verifier.timeout_sec
    )

    duration_ms = int((time.time() - start) * 1000)

    passed = result.returncode == 0

    score = 1.0 if passed else 0.0

    return VerifierResult(
        task_id=task.task_id,
        passed=passed,
        score=score,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_ms=duration_ms
    )