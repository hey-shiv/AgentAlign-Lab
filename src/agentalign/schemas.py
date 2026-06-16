"""Pydantic v2 schemas for the AgentAlign Lab pipeline.

Defines all core data models used throughout the project:
tasks, actions, steps, trajectories, verifier results, and preference pairs.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Task schemas
# ---------------------------------------------------------------------------

class TaskFile(BaseModel):
    """A single file to be placed in the agent's workspace."""
    path: str
    content: str


class VerifierConfig(BaseModel):
    """Configuration for the deterministic verifier that grades agent output."""
    type: str = "pytest"
    command: str = ""
    timeout_sec: int = 8
    expected_output: Optional[str] = None
    # Extended fields used by specific verifier types
    target_path: Optional[str] = None
    expected_content: Optional[str] = None
    expected_json: Any = None
    required_fields: list[str] = Field(default_factory=list)


# Backward-compatible alias
TaskVerifier = VerifierConfig


DEFAULT_FORBIDDEN = ["rm", "curl", "wget", "pip", "sudo", "chmod", "ssh", "git"]


class Task(BaseModel):
    """A self-contained coding task for the agent to solve."""
    model_config = ConfigDict(populate_by_name=True)

    task_id: str
    family: str = "python_bugfix"
    difficulty: Optional[str] = None
    instruction: str = Field(default="", validation_alias="description")
    files: list[TaskFile] = Field(default_factory=list)
    verifier: VerifierConfig = Field(
        default_factory=lambda: VerifierConfig(type="pytest", command="pytest -q")
    )
    forbidden_commands: list[str] = Field(default_factory=lambda: list(DEFAULT_FORBIDDEN))

    @field_validator("task_id")
    @classmethod
    def _task_id_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task_id must not be empty")
        return v

    @field_validator("instruction")
    @classmethod
    def _instruction_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("instruction must not be empty")
        return v

    @property
    def description(self) -> str:
        """Alias kept for backward compatibility."""
        return self.instruction


# ---------------------------------------------------------------------------
# Agent action / step schemas
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"list_files", "read_file", "write_file", "run_command", "final_answer"}


class Action(BaseModel):
    """A single action proposed by the agent."""
    thought: str = ""
    action: str
    args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _compat(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Support legacy 'name' field
            if "name" in data and "action" not in data:
                data = {**data, "action": data.pop("name")}
            if "thought" not in data:
                data = {**data, "thought": ""}
        return data

    # Keep legacy .name property
    @property
    def name(self) -> str:
        return self.action


class Step(BaseModel):
    """A single step in an agent trajectory."""
    model_config = ConfigDict(populate_by_name=True)

    step_index: int = Field(validation_alias="step_id")
    thought: str = ""
    action: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    observation: Optional[str] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None

    @property
    def step_id(self) -> int:
        """Legacy alias."""
        return self.step_index


# ---------------------------------------------------------------------------
# Verifier result
# ---------------------------------------------------------------------------

class VerifierResult(BaseModel):
    """Result of running the deterministic verifier on an agent workspace."""
    task_id: str = ""
    passed: bool = False
    score: float = 0.0
    num_steps: int = 0
    failed_commands: int = 0
    unsafe_actions: int = 0
    invalid_actions: int = 0
    partial_credits: list[str] = Field(default_factory=list)
    failure_label: Optional[str] = None
    # Extended fields
    failure_tags: list[str] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0

    @model_validator(mode="before")
    @classmethod
    def _handle_success_alias(cls, data: Any) -> Any:
        """Accept 'success' as alias for 'passed'."""
        if isinstance(data, dict) and "success" in data and "passed" not in data:
            data = {**data, "passed": data.pop("success")}
        return data

    @property
    def success(self) -> bool:
        """Legacy alias."""
        return self.passed


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

class Trajectory(BaseModel):
    """Complete record of one agent run on one task."""
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(validation_alias="trajectory_id")
    task_id: str
    agent_id: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    steps: list[Step] = Field(default_factory=list)
    final_answer: Optional[str] = None
    verifier_result: Optional[VerifierResult] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _legacy_success_score(cls, data: Any) -> Any:
        """Convert legacy top-level success/score into a VerifierResult."""
        if not isinstance(data, dict) or data.get("verifier_result") is not None:
            return data
        if "success" in data or "score" in data:
            data = dict(data)
            task_id = data.get("task_id", "")
            data["verifier_result"] = {
                "task_id": task_id,
                "passed": bool(data.pop("success", False)),
                "score": float(data.pop("score", 0.0)),
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


# ---------------------------------------------------------------------------
# Preference pair schemas
# ---------------------------------------------------------------------------

class PreferencePair(BaseModel):
    """A DPO preference pair: chosen trajectory vs rejected trajectory."""
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
    """Simplified preference triple for backward compatibility."""
    prompt: str
    chosen: str
    rejected: str
