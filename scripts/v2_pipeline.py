#!/usr/bin/env python3
"""AgentAlign v2 pipeline CLI.

Stages (each resumable, GPU stages shardable across GPUs with --shard/--nshards):
  rollout  batched episodes on a task split (used for both training data and evaluation)
  states   pick clean successful trajectories and turn their good steps into states
  alts     sample alternative actions at each state
  build    SFT examples + verifier-labelled DPO pairs, split by task
  train    SFT or DPO (QLoRA on CUDA)
  merge    merge a LoRA adapter into an fp16 model directory
  report   metrics, paired bootstrap CIs, markdown table
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentalign.schemas import Trajectory  # noqa: E402
from agentalign.tasks.load import load_tasks_from_jsonl  # noqa: E402
from agentalign.v2 import data as v2data  # noqa: E402


def _tasks(split: str, max_tasks: int | None = None):
    tasks = load_tasks_from_jsonl(ROOT / "data" / "tasks" / f"{split}.jsonl")
    if not tasks:
        raise SystemExit(f"no tasks found for split '{split}' in data/tasks/{split}.jsonl")
    return tasks[:max_tasks] if max_tasks else tasks


def _seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _policy(args, temperature: float, top_p: float):
    if args.model == "scripted":
        # Deterministic CPU policy for tests: reuses the v1 scripted baseline.
        import importlib.util

        spec = importlib.util.spec_from_file_location("run_agent", ROOT / "scripts" / "02_run_agent.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return lambda prompts: [mod._baseline_callable(p) for p in prompts]
    from agentalign.v2.hf_policy import HFPolicy

    return HFPolicy(
        args.model, adapter_path=args.adapter, temperature=temperature, top_p=top_p,
        max_new_tokens=args.max_new_tokens, token_budget=args.token_budget, max_batch=args.max_batch,
    )


def cmd_rollout(args) -> None:
    from agentalign.v2.engine import run_wave

    tasks = _tasks(args.split, args.max_tasks)
    jobs = [(t, s) for t in tasks for s in range(args.samples)][args.shard::args.nshards]
    out = Path(args.out) / f"shard{args.shard}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["task_id"], r["metadata"]["sample_idx"]) for r in v2data.read_jsonl(out)}
    todo = [(t, s) for t, s in jobs if (t.task_id, s) not in done]
    print(f"[rollout {args.agent_id} shard {args.shard}/{args.nshards}] {len(jobs)} jobs, {len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return
    policy = _policy(args, args.temperature, args.top_p)
    model_name = args.model + (f"+{args.adapter}" if args.adapter else "")
    start = time.time()
    finished = passed = 0
    for w in range(0, len(todo), args.wave_size):
        if args.deadline and time.time() > args.deadline:
            print(f"[rollout] deadline reached; stopping with {len(todo) - finished} jobs left (rerun to resume)", flush=True)
            break
        _seed(args.seed + args.shard * 100_000 + w)
        wave = todo[w:w + args.wave_size]
        trajs = run_wave(wave, policy, args.agent_id, model_name, max_steps=args.max_steps,
                         temperature=args.temperature, tool_workers=args.tool_workers)
        with out.open("a") as f:
            for t in trajs:
                t.seed = args.seed
                f.write(t.model_dump_json() + "\n")
        finished += len(trajs)
        passed += sum(1 for t in trajs if t.verifier_result and t.verifier_result.passed)
        rate = (time.time() - start) / finished
        print(f"[rollout {args.agent_id} shard {args.shard}] {finished}/{len(todo)} done, pass {passed}/{finished}, "
              f"{rate:.1f}s/episode, ETA {rate * (len(todo) - finished) / 60:.1f} min", flush=True)


def _load_dir(path: str) -> list[Trajectory]:
    return v2data.load_trajectories(sorted(Path(path).rglob("*.jsonl")))


def cmd_states(args) -> None:
    trajs = _load_dir(args.rollouts)
    successes = v2data.select_successes(trajs, args.max_success_per_task, seed=args.seed)
    states = v2data.build_states(successes)
    n = v2data.write_jsonl(states, args.out)
    tasks_any = len({t.task_id for t in trajs})
    tasks_succ = len({t.task_id for t in successes})
    clean = sum(1 for t in trajs if v2data.is_clean_success(t))
    summary = {"rollouts": len(trajs), "clean_successes": clean, "tasks": tasks_any,
               "tasks_with_success": tasks_succ, "successes_used": len(successes), "states": n}
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_alts(args) -> None:
    states = v2data.read_jsonl(args.states)[args.shard::args.nshards]
    out = Path(args.out) / f"shard{args.shard}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["state_id"] for r in v2data.read_jsonl(out)}
    todo = [s for s in states if s["state_id"] not in done]
    print(f"[alts shard {args.shard}/{args.nshards}] {len(states)} states, {len(todo)} to sample x{args.k}", flush=True)
    if not todo:
        return
    policy = _policy(args, args.temperature, args.top_p)
    start = time.time()
    for w in range(0, len(todo), args.wave_size):
        if args.deadline and time.time() > args.deadline:
            print("[alts] deadline reached; stopping (rerun to resume)", flush=True)
            break
        _seed(args.seed + args.shard * 100_000 + w)
        chunk = todo[w:w + args.wave_size]
        prompts = [s["prompt"] for s in chunk for _ in range(args.k)]
        outs = policy(prompts)
        with out.open("a") as f:
            for i, s in enumerate(chunk):
                f.write(json.dumps({"state_id": s["state_id"], "alternatives": outs[i * args.k:(i + 1) * args.k]}) + "\n")
        print(f"[alts shard {args.shard}] {min(w + args.wave_size, len(todo))}/{len(todo)} states, "
              f"{(time.time() - start) / 60:.1f} min", flush=True)


def cmd_build(args) -> None:
    states = v2data.read_jsonl(args.states)
    alts: dict[str, list[str]] = {}
    for p in sorted(Path(args.alts).glob("*.jsonl")) if args.alts else []:
        for r in v2data.read_jsonl(p):
            alts[r["state_id"]] = r["alternatives"]
    tasks = {t.task_id: t for t in _tasks(args.split)}
    sft = v2data.build_sft_examples(states)
    pairs = v2data.build_dpo_pairs(states, alts, tasks, max_pairs_per_state=args.max_pairs_per_state)
    if args.smoke_fallback and (not sft or not pairs):
        # Smoke runs only: exercise the training code even if a tiny rollout had no successes.
        # These rows are not used for any reported result.
        trajs = _load_dir(args.rollouts) if args.rollouts else []
        fallback = []
        for t in trajs:
            for row in t.metadata.get("step_io", [])[:1]:
                fallback.append({"task_id": t.task_id, "prompt": row["prompt"],
                                 "completion": '{"thought": "List the workspace first.", "action": "list_files", "args": {}}'})
        sft = sft or fallback
        pairs = pairs or [{"pair_id": f"smoke:{i}", "task_id": r["task_id"], "step_index": 1, "prompt": r["prompt"],
                           "chosen": r["completion"], "rejected": "I will just guess.", "reject_reason": "invalid_json",
                           "source": "smoke_fallback"} for i, r in enumerate(fallback)]
        print(f"[build] smoke fallback: sft={len(sft)} pairs={len(pairs)}")
    rng = random.Random(args.seed)
    if args.max_dpo_pairs and len(pairs) > args.max_dpo_pairs:
        rng.shuffle(pairs)
        pairs = sorted(pairs[:args.max_dpo_pairs], key=lambda p: p["pair_id"])
    # One task-level split shared by SFT and DPO, so neither sees the held-out tasks.
    all_tasks = sorted({r["task_id"] for r in sft} | {r["task_id"] for r in pairs})
    _, held_rows = v2data.split_by_task([{"task_id": t} for t in all_tasks], args.eval_fraction, seed=args.seed)
    held = {r["task_id"] for r in held_rows}
    out = Path(args.out_dir)
    counts = {
        "sft_train": v2data.write_jsonl([r for r in sft if r["task_id"] not in held], out / "sft_train.jsonl"),
        "sft_eval": v2data.write_jsonl([r for r in sft if r["task_id"] in held], out / "sft_eval.jsonl"),
        "dpo_train": v2data.write_jsonl([r for r in pairs if r["task_id"] not in held], out / "dpo_train.jsonl"),
        "dpo_eval": v2data.write_jsonl([r for r in pairs if r["task_id"] in held], out / "dpo_eval.jsonl"),
    }
    summary = {
        **counts,
        "states": len(states),
        "states_with_alternatives": sum(1 for s in states if s["state_id"] in alts),
        "held_out_tasks": sorted(held),
        "reject_reasons": dict(Counter(p["reject_reason"] for p in pairs)),
        "pairs_by_family": dict(Counter(p["task_id"].rsplit("_", 1)[0] for p in pairs)),
    }
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_train(args) -> None:
    from agentalign.v2.train import train_dpo, train_sft

    cfg = json.loads(Path(args.config).read_text())
    train_rows = v2data.read_jsonl(args.train)
    eval_rows = v2data.read_jsonl(args.eval) if args.eval else []
    fn = train_sft if args.kind == "sft" else train_dpo
    meta = fn(args.model, train_rows, eval_rows, cfg, args.out)
    print(json.dumps({k: v for k, v in meta.items() if k not in {"log_history", "config"}}, indent=2, default=str))


def cmd_merge(args) -> None:
    from agentalign.v2.train import merge_adapter

    print(merge_adapter(args.base, args.adapter, args.out))


def cmd_report(args) -> None:
    from agentalign.v2.metrics import arm_metrics, paired_bootstrap, unsafe_key

    arms: dict[str, list[Trajectory]] = defaultdict(list)
    for t in _load_dir(args.eval_dir):
        arms[t.agent_id].append(t)
    order = [a for a in args.arms.split(",") if a in arms] + sorted(set(arms) - set(args.arms.split(",")))
    result = {"arms": {a: arm_metrics(arms[a]) for a in order}, "comparisons": {}}
    for a, b in [(x, y) for i, x in enumerate(order) for y in order[i + 1:]]:
        result["comparisons"][f"{b}_vs_{a}"] = {
            "pass_rate": paired_bootstrap(arms[a], arms[b]),
            "runs_with_unsafe": paired_bootstrap(arms[a], arms[b], key=unsafe_key),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    lines = ["| Arm | Pass rate | Runs with unsafe cmd | Invalid actions / step | Avg steps | Runs |",
             "|---|---|---|---|---|---|"]
    for a in order:
        m = result["arms"][a]
        lines.append(f"| {a} | {m['pass_rate']*100:.1f}% ({m['passed']}/{m['runs']}) | {m['runs_with_unsafe_rate']*100:.1f}% | "
                     f"{m['invalid_actions_per_step']:.3f} | {m['avg_steps']:.2f} | {m['runs']} |")
    lines += ["", "| Comparison | Pass-rate diff | 95% CI | p | Unsafe-run diff | 95% CI |", "|---|---|---|---|---|---|"]
    for k, c in result["comparisons"].items():
        p, u = c["pass_rate"], c["runs_with_unsafe"]
        lines.append(f"| {k} | {p['diff']*100:+.1f} pts | [{p['ci_lo']*100:+.1f}, {p['ci_hi']*100:+.1f}] | {p['p_value']:.3f} | "
                     f"{u['diff']*100:+.1f} pts | [{u['ci_lo']*100:+.1f}, {u['ci_hi']*100:+.1f}] |")
    md = "\n".join(lines)
    Path(args.out).with_suffix(".md").write_text(md + "\n")
    print(md)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def gen_args(sp, temperature: float, top_p: float):
        sp.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
        sp.add_argument("--adapter", default=None)
        sp.add_argument("--temperature", type=float, default=temperature)
        sp.add_argument("--top-p", type=float, default=top_p)
        sp.add_argument("--max-new-tokens", type=int, default=512)
        sp.add_argument("--token-budget", type=int, default=120_000)
        sp.add_argument("--max-batch", type=int, default=64)
        sp.add_argument("--wave-size", type=int, default=128)
        sp.add_argument("--shard", type=int, default=0)
        sp.add_argument("--nshards", type=int, default=1)
        sp.add_argument("--deadline", type=float, default=0.0, help="unix time after which no new wave starts")
        sp.add_argument("--seed", type=int, default=42)

    r = sub.add_parser("rollout")
    gen_args(r, 0.8, 1.0)
    r.add_argument("--split", required=True)
    r.add_argument("--samples", type=int, required=True)
    r.add_argument("--agent-id", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-steps", type=int, default=8)
    r.add_argument("--max-tasks", type=int, default=None)
    r.add_argument("--tool-workers", type=int, default=8)
    r.set_defaults(fn=cmd_rollout)

    s = sub.add_parser("states")
    s.add_argument("--rollouts", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-success-per-task", type=int, default=4)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_states)

    a = sub.add_parser("alts")
    gen_args(a, 1.0, 1.0)
    a.add_argument("--states", required=True)
    a.add_argument("--k", type=int, default=4)
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_alts)

    b = sub.add_parser("build")
    b.add_argument("--states", required=True)
    b.add_argument("--alts", default=None)
    b.add_argument("--split", default="train")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--eval-fraction", type=float, default=0.1)
    b.add_argument("--max-pairs-per-state", type=int, default=2)
    b.add_argument("--max-dpo-pairs", type=int, default=0)
    b.add_argument("--smoke-fallback", action="store_true")
    b.add_argument("--rollouts", default=None, help="rollout dir, only used by --smoke-fallback")
    b.add_argument("--seed", type=int, default=42)
    b.set_defaults(fn=cmd_build)

    t = sub.add_parser("train")
    t.add_argument("--kind", choices=["sft", "dpo"], required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--train", required=True)
    t.add_argument("--eval", default=None)
    t.add_argument("--config", required=True)
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_train)

    m = sub.add_parser("merge")
    m.add_argument("--base", required=True)
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_merge)

    rp = sub.add_parser("report")
    rp.add_argument("--eval-dir", required=True)
    rp.add_argument("--arms", default="qwen_base,qwen_sft,qwen_sft_dpo")
    rp.add_argument("--out", required=True)
    rp.set_defaults(fn=cmd_report)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
