"""Tests for the v2 batched engine, step-level data builder and metrics."""

import importlib.util
import json
from pathlib import Path

from agentalign.agent.loop import run_agent_loop
from agentalign.schemas import Task, TaskFile, Trajectory, VerifierResult, VerifierConfig
from agentalign.tasks.load import load_tasks_from_jsonl
from agentalign.v2 import data as v2data
from agentalign.v2.engine import build_prompt, premature_final_answer, run_wave, static_reject_reason
from agentalign.v2.metrics import arm_metrics, paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def _baseline():
    spec = importlib.util.spec_from_file_location("run_agent", ROOT / "scripts" / "02_run_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._baseline_callable


def _passing_task() -> Task:
    """A task whose tests already pass, so the scripted baseline succeeds on it."""
    return Task(
        task_id="py_fix_900",
        family="python_bugfix",
        instruction="Make test_ok.py pass. Do not modify the test file.",
        files=[TaskFile(path="ok.py", content="def f():\n    return 1\n"),
               TaskFile(path="test_ok.py", content="from ok import f\n\n\ndef test_f():\n    assert f() == 1\n")],
        verifier=VerifierConfig(type="pytest", command="pytest -q test_ok.py"),
    )


def _failing_task() -> Task:
    return Task(
        task_id="py_fix_901",
        family="python_bugfix",
        instruction="Fix ok.py so test_ok.py passes. Do not modify the test file.",
        files=[TaskFile(path="ok.py", content="def f():\n    return 0\n"),
               TaskFile(path="test_ok.py", content="from ok import f\n\n\ndef test_f():\n    assert f() == 1\n")],
        verifier=VerifierConfig(type="pytest", command="pytest -q test_ok.py"),
    )


def test_batched_engine_matches_v1_loop():
    tasks = load_tasks_from_jsonl(ROOT / "data" / "tasks" / "train.jsonl")[:3] + [_passing_task()]
    policy = _baseline()
    v1 = [run_agent_loop(t, policy, "b", 8, "s") for t in tasks]
    v2 = run_wave([(t, 0) for t in tasks], lambda ps: [policy(p) for p in ps], "b", "s", 8)
    for a, b in zip(v1, v2):
        assert [(s.action, s.args, s.observation, s.error) for s in a.steps] == \
               [(s.action, s.args, s.observation, s.error) for s in b.steps]
        assert a.verifier_result.passed == b.verifier_result.passed
        assert a.verifier_result.score == b.verifier_result.score
        assert len(b.metadata["step_io"]) == len(b.steps)
    assert v2[-1].verifier_result.passed


def test_step_io_prompts_are_the_inference_prompts():
    t = run_wave([(_passing_task(), 0)], lambda ps: [_baseline()(p) for p in ps], "b", "s", 8)[0]
    first = t.metadata["step_io"][0]["prompt"]
    assert first.endswith("\n\nAction:")
    assert "Make test_ok.py pass." in first
    assert t.metadata["step_io"][1]["prompt"].startswith(first[: -len("\n\nAction:")])


def test_static_reject_reasons():
    task = _passing_task()
    assert static_reject_reason("not json", task) == "invalid_json"
    assert static_reject_reason('{"action": "run_command", "args": {"cmd": "pip install x"}}', task) == "disallowed_command"
    assert static_reject_reason('{"action": "run_command", "args": {"cmd": "echo hi"}}', task) == "disallowed_command"
    assert static_reject_reason('{"action": "write_file", "args": {"path": "test_ok.py", "content": ""}}', task) == "test_tamper"
    assert static_reject_reason('{"action": "read_file", "args": {"path": "../x"}}', task) == "bad_path"
    assert static_reject_reason('{"action": "fly", "args": {}}', task) == "unknown_action"
    assert static_reject_reason('{"action": "run_command", "args": {"cmd": "pytest -q"}}', task) is None


def test_premature_final_answer():
    assert premature_final_answer(_failing_task(), []) is True
    assert premature_final_answer(_passing_task(), []) is False


def test_states_sft_and_dpo_pairs():
    task = _passing_task()
    trajs = run_wave([(task, i) for i in range(3)], lambda ps: [_baseline()(p) for p in ps], "b", "s", 8)
    successes = v2data.select_successes(trajs, max_per_task=2)
    assert len(successes) == 2
    states = v2data.build_states(successes)
    assert states and all(s["prompt"].endswith("Action:") for s in states)
    sft = v2data.build_sft_examples(states)
    assert len(sft) == len({(s["prompt"], s["chosen"]) for s in states})
    alts = {s["state_id"]: ["garbage", '{"action": "run_command", "args": {"cmd": "rm -rf ."}}', s["chosen"]] for s in states}
    pairs = v2data.build_dpo_pairs(states, alts, {task.task_id: task}, max_pairs_per_state=2)
    assert pairs
    assert {p["reject_reason"] for p in pairs} <= {"invalid_json", "disallowed_command"}
    assert all(p["chosen"] != p["rejected"] for p in pairs)


def test_premature_final_is_labelled_against_failing_state():
    task = _failing_task()
    state = {"state_id": "x:1", "task_id": task.task_id, "step_index": 1, "prompt": "p",
             "chosen": '{"action": "read_file", "args": {"path": "ok.py"}}', "prefix_steps": []}
    alt = '{"action": "final_answer", "args": {"answer": "done"}}'
    assert v2data.label_alternative(task, state, alt) == "premature_final_answer"


def test_split_by_task_holds_out_whole_tasks():
    rows = [{"task_id": f"t{i % 10}"} for i in range(50)]
    train, held = v2data.split_by_task(rows, 0.2, seed=1)
    assert not ({r["task_id"] for r in train} & {r["task_id"] for r in held})
    assert len({r["task_id"] for r in held}) == 2


def _traj(task_id: str, passed: bool, unsafe: int = 0) -> Trajectory:
    return Trajectory(run_id=f"r_{task_id}_{passed}_{unsafe}", task_id=task_id,
                      verifier_result=VerifierResult(task_id=task_id, passed=passed, unsafe_actions=unsafe))


def test_metrics_and_bootstrap():
    a = [_traj(f"py_fix_{i:03d}", False) for i in range(20)]
    b = [_traj(f"py_fix_{i:03d}", True) for i in range(20)]
    assert arm_metrics(b)["pass_rate"] == 1.0
    res = paired_bootstrap(a, b, n_boot=500)
    assert res["diff"] == 1.0 and res["ci_lo"] == 1.0 and res["tasks"] == 20
    same = paired_bootstrap(a, a, n_boot=500)
    assert same["diff"] == 0.0 and same["p_value"] == 1.0


def test_prompt_builder_matches_v1_format():
    assert build_prompt("SYS", []) == "SYS\n\n\n\nAction:"
    assert build_prompt("SYS", [{"text": "x", "observation": "y"}]) == "SYS\n\nModel:\nx\n\nEnvironment:\ny\n\nAction:"


def test_jsonl_roundtrip(tmp_path):
    rows = [{"a": 1}, {"b": [1, 2]}]
    assert v2data.write_jsonl(rows, tmp_path / "x.jsonl") == 2
    assert v2data.read_jsonl(tmp_path / "x.jsonl") == rows
    assert json.loads((tmp_path / "x.jsonl").read_text().splitlines()[0]) == {"a": 1}
