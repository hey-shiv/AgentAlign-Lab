from pydantic import BaseModel


class TaskFile(BaseModel):
    path: str
    content: str


class VerifierConfig(BaseModel):
    type: str
    command: str
    timeout_sec: int


class Task(BaseModel):
    task_id: str
    family: str
    instruction: str
    files: list[TaskFile]
    verifier: VerifierConfig


class VerifierResult(BaseModel):
    task_id: str
    passed: bool
    score: float
    stdout: str
    stderr: str
    duration_ms: int