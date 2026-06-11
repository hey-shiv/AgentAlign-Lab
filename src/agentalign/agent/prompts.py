from agentalign.schemas import Task


def build_system_prompt(task: Task) -> str:
    return f"""You are a terminal-based AI agent.
Your task is to solve the following instruction by executing commands, reading files, and writing files.

INSTRUCTION:
{task.instruction}

AVAILABLE ACTIONS:
- list_files: {{}}
- read_file: {{"path": "<file_path>"}}
- write_file: {{"path": "<file_path>", "content": "<new_content>"}}
- run_command: {{"cmd": "<command_string>"}}
- final_answer: {{"answer": "<summary_of_fix>"}}

You must ALWAYS respond with a SINGLE valid JSON object in the following format:
{{
    "thought": "Your reasoning about what to do next",
    "action": "action_name",
    "args": {{...}}
}}

RULES:
1. Always run tests using `run_command` to verify your fix before calling `final_answer`.
2. Do not use interactive commands like `vim`, `nano`, or `less`. Use `cat` or `read_file` instead.
3. Path must stay inside the workspace.
4. Output only JSON, no markdown blocks around it.
"""

def format_history(history: list[dict]) -> str:
    res = []
    for h in history:
        res.append(f"Model:\n{h['text']}")
        if h.get('error'):
            res.append(f"Environment Error:\n{h['error']}")
        elif h.get('observation'):
            res.append(f"Environment:\n{h['observation']}")
    return "\n\n".join(res)
