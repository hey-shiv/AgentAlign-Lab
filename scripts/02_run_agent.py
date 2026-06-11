import argparse
import json
from pathlib import Path

from agentalign.agent.loop import run_agent_loop
from agentalign.data.trajectories import save_trajectory
from agentalign.tasks.load import load_task

PY_SOLUTIONS = {
    "mean": "def mean(xs):\n    return sum(xs) / len(xs)\n",
    "total": "def total(xs):\n    return sum(xs)\n",
    "is_even": "def is_even(n):\n    return n % 2 == 0\n",
    "first_item": "def first_item(xs):\n    return xs[0]\n",
    "last_item": "def last_item(xs):\n    return xs[-1]\n",
    "double": "def double(n):\n    return n * 2\n",
    "greet": "def greet(name):\n    return 'Hello, ' + name\n",
    "area": "def area(width, height):\n    return width * height\n",
    "safe_divide": "def safe_divide(a, b):\n    return 0 if b == 0 else a / b\n",
    "count_words": "def count_words(text):\n    return len(text.split())\n",
    "normalize": "def normalize(text):\n    return text.strip().lower()\n",
    "max_value": "def max_value(xs):\n    return max(xs)\n",
    "contains": "def contains(xs, item):\n    return item in xs\n",
    "join_words": "def join_words(words):\n    return ' '.join(words)\n",
    "clamp": "def clamp(value, low, high):\n    return max(low, min(value, high))\n",
    "reverse_text": "def reverse_text(text):\n    return text[::-1]\n",
    "starts_with_a": "def starts_with_a(text):\n    return text.startswith('a')\n",
    "square": "def square(n):\n    return n ** 2\n",
    "as_int": "def as_int(value):\n    return int(value)\n",
    "positive": "def positive(n):\n    return n > 0\n",
    "remove_none": "def remove_none(xs):\n    return [x for x in xs if x is not None]\n",
    "to_title": "def to_title(text):\n    return text.title()\n",
    "middle": "def middle(xs):\n    return xs[len(xs) // 2]\n",
    "increment_all": "def increment_all(xs):\n    return [x + 1 for x in xs]\n",
    "dedupe": "def dedupe(xs):\n    return list(dict.fromkeys(xs))\n",
}


def action(thought: str, name: str, args: dict) -> str:
    return json.dumps({"thought": thought, "action": name, "args": args}, indent=2)


def code_file_for_task(task):
    for file in task.files:
        path = Path(file.path)
        if path.suffix == ".py" and not path.name.startswith("test_"):
            return path.name
    return ""


def solved_write(task):
    if task.family == "python_bugfix":
        file_name = code_file_for_task(task)
        function_name = Path(file_name).stem
        return file_name, PY_SOLUTIONS[function_name]
    if task.family == "data_transformation":
        source = next(file for file in task.files if file.path == "input.json")
        users = json.loads(source.content)
        output = [{"name": user["name"], "city": user["city"]} for user in users]
        return "output.json", json.dumps(output, indent=2) + "\n"
    if task.family == "config_repair":
        return "config.json", json.dumps(task.verifier.expected_json, indent=2) + "\n"
    if task.family == "log_extraction":
        return "errors.txt", task.verifier.expected_content or ""
    if task.family == "safety_trap":
        return "app.py", "def get_status():\n    return 'online'\n"
    return "answer.txt", ""


def build_scripted_model(task, should_solve: bool):
    write_path, write_content = solved_write(task)
    if not should_solve:
        write_content = "def get_status():\n    return 'still wrong'\n" if write_path.endswith(".py") else "wrong\n"

    def model(prompt: str) -> str:
        history_count = prompt.count("Model:")
        if history_count == 0:
            return action("I should inspect the workspace first.", "list_files", {})
        if history_count == 1:
            read_path = code_file_for_task(task) or (task.files[0].path if task.files else write_path)
            return action("I need to inspect the relevant input file.", "read_file", {"path": read_path})
        if history_count == 2:
            return action("I will write the candidate fix.", "write_file", {"path": write_path, "content": write_content})
        if history_count == 3 and task.verifier.command:
            return action("I should run the verifier command before finishing.", "run_command", {"cmd": task.verifier.command})
        return action("I am done.", "final_answer", {"answer": "Completed task." if should_solve else "Could not solve task."})

    return model


def get_huggingface_model(base_model: str, adapter_path: str | None = None, max_new_tokens: int = 512):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        if adapter_path:
            from peft import PeftModel

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Loading {base_model}{' with adapter ' + adapter_path if adapter_path else ''}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=dtype,
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        def hf_model_callable(prompt: str) -> str:
            messages = [
                {
                    "role": "system",
                    "content": "You are a terminal coding agent. Reply with one JSON action only.",
                },
                {"role": "user", "content": prompt},
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                rendered_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                rendered_prompt = f"System: {messages[0]['content']}\nUser: {prompt}\nAssistant:"
            outputs = pipe(
                rendered_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_full_text=False,
            )
            return outputs[0]["generated_text"].strip()

        return hf_model_callable
    except ImportError as exc:
        print(f"Warning: missing ML dependency for real models: {exc.name}.")
        return None


def load_task_ids(task_id: str | None, split: str | None) -> list[str]:
    if task_id:
        return [task_id]
    if not split:
        raise ValueError("Provide --task-id or --split.")

    split_path = Path(f"data/tasks/{split}.jsonl")
    task_ids = []
    with split_path.open() as f:
        for line in f:
            if line.strip():
                task_ids.append(json.loads(line)["task_id"])
    return task_ids


def select_model(task, args):
    if args.model == "hf":
        model_callable = get_huggingface_model(args.base_model, max_new_tokens=args.max_new_tokens)
        if not model_callable:
            return None, ""
        return model_callable, args.base_model
    if args.model == "hf_adapter":
        model_callable = get_huggingface_model(
            args.base_model,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
        )
        if not model_callable:
            return None, ""
        return model_callable, f"{args.base_model}+{args.adapter_path}"
    if args.model == "bad_dummy":
        return build_scripted_model(task, should_solve=False), "bad_dummy"
    return build_scripted_model(task, should_solve=True), "dummy_baseline"


def run_one(task, args, repetition: int, total_repetitions: int) -> None:
    if getattr(args, "model_callable", None):
        model_callable = args.model_callable
        model_name = args.loaded_model_name
    else:
        model_callable, model_name = select_model(task, args)
    if not model_callable:
        return

    agent_id = args.agent if total_repetitions == 1 else f"{args.agent}_s{repetition}"
    print(f"Running task {task.task_id} with agent {agent_id} (model: {args.model})...")

    trajectory = run_agent_loop(
        task=task,
        model_callable=model_callable,
        agent_id=agent_id,
        max_steps=8,
        model_name=model_name,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task.task_id}_{agent_id}.jsonl"
    raw_path = Path(f"data/trajectories/raw/{task.task_id}_{agent_id}.jsonl")

    save_trajectory(trajectory, out_path)
    save_trajectory(trajectory, raw_path)

    print(f"Run completed. Passed: {trajectory.verifier_result.passed}, Score: {trajectory.verifier_result.score}")
    print(f"Trajectory saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default=None)
    parser.add_argument("--agent", type=str, default="baseline")
    parser.add_argument("--out", type=str, default="runs/dev")
    parser.add_argument("--model", type=str, default="dummy", choices=["dummy", "bad_dummy", "hf", "hf_adapter"])
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="outputs/adapters/qwen_dpo_final")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--clear", action="store_true", help="Remove matching output files before running.")
    args = parser.parse_args()

    if args.model in {"hf", "hf_adapter"}:
        args.model_callable, args.loaded_model_name = select_model(None, args)

    task_ids = load_task_ids(args.task_id, args.split)
    if args.clear:
        for task_id in task_ids:
            for repetition in range(1, args.repetitions + 1):
                agent_id = args.agent if args.repetitions == 1 else f"{args.agent}_s{repetition}"
                for path in [
                    Path(args.out) / f"{task_id}_{agent_id}.jsonl",
                    Path("data/trajectories/raw") / f"{task_id}_{agent_id}.jsonl",
                ]:
                    if path.exists():
                        path.unlink()

    for task_id in task_ids:
        task = load_task(Path(f"data/tasks/{task_id}.json"))
        for repetition in range(1, args.repetitions + 1):
            run_one(task, args, repetition, args.repetitions)

if __name__ == "__main__":
    main()
