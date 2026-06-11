from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskFile(BaseModel):
    path: str
    content: str

class VerifierConfig(BaseModel):
    type: str
    command: str = ""
    timeout_sec: int = 8
    target_path: str | None = None
    expected_content: str | None = None
    expected_json: Any | None = None
    required_fields: list[str] = Field(default_factory=list)

class Task(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str
    family: str = "general"
    difficulty: str | None = None
    instruction: str = Field(default="", validation_alias="description")
    files: list[TaskFile] = Field(default_factory=list)
    verifier: VerifierConfig = Field(
        default_factory=lambda: VerifierConfig(type="none", command="true", timeout_sec=1)
    )
    forbidden_commands: list[str] = Field(default_factory=list)

    @field_validator("task_id", "instruction")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @property
    def description(self) -> str:
        return self.instruction

class Action(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)

class Step(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    step_index: int = Field(validation_alias="step_id")
    thought: str = ""
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None
    latency_ms: int | None = None
    error: str | None = None

    @property
    def step_id(self) -> int:
        return self.step_index

class VerifierResult(BaseModel):
    task_id: str
    passed: bool
    score: float
    failure_tags: list[str] = Field(default_factory=list)
    stdout: str
    stderr: str
    duration_ms: int
    num_steps: int = 0
    failed_commands: int = 0
    unsafe_actions: int = 0
    invalid_actions: int = 0

class Trajectory(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(validation_alias="trajectory_id")
    task_id: str
    agent_id: str | None = None
    model: str | None = None
    temperature: float | None = None
    seed: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    steps: list[Step] = Field(default_factory=list)
    final_answer: str | None = None
    verifier_result: VerifierResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _legacy_success_score(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("verifier_result") is not None:
            return data

        if "success" in data or "score" in data:
            data = data.copy()
            task_id = data.get("task_id", "")
            data["verifier_result"] = {
                "task_id": task_id,
                "passed": bool(data.pop("success", False)),
                "score": float(data.pop("score", 0.0)),
                "failure_tags": [],
                "stdout": "",
                "stderr": "",
                "duration_ms": 0,
            }
        return data

    @property
    def trajectory_id(self) -> str:
        return self.run_id

    @property
    def success(self) -> bool:
        return bool(self.verifier_result and self.verifier_result.passed)

    @property
    def score(self) -> float:
        if not self.verifier_result:
            return 0.0
        return self.verifier_result.score

class PreferencePair(BaseModel):
    pair_id: str
    task_id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    score_margin: float
    source: str = "deterministic_verifier"


class Preference(BaseModel):
    prompt: str
    chosen: str
    rejected: str
