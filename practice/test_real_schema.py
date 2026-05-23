from agentalign.schemas import (
    Task,
    TaskFile,
    VerifierConfig,
)


task = Task(
    task_id="py_fix_001",
    family="python_bugfix",
    instruction="Fix stats.py so tests pass.",
    files=[
        TaskFile(
            path="stats.py",
            content="def mean(xs): return sum(xs)/len(xs)+1"
        ),
        TaskFile(
            path="test_stats.py",
            content="""
from stats import mean

def test_mean():
    assert mean([1,2,3]) == 2
"""
        )
    ],
    verifier=VerifierConfig(
        type="pytest",
        command="pytest -q",
        timeout_sec=10
    )
)

print(task)

print(type(task.files[0]))
print(type(task.verifier))