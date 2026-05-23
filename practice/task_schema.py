from pydantic import BaseModel


class VerifierConfig(BaseModel):
    type: str
    command: str
    timeout_sec: int


class Task(BaseModel):
    task_id: str
    instruction: str
    verifier: VerifierConfig


task = Task(
    task_id="py_fix_001",
    instruction="Fix stats.py",
    verifier={
        "type": "pytest",
        "command": "pytest -q",
        "timeout_sec": 10
    }
)

print(task)
print(type(task.verifier))