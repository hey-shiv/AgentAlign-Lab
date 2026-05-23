from agentalign.loader import load_task
from agentalign.workspace import create_workspace
from agentalign.verifier import run_verifier


task = load_task("data/tasks/py_fix_001.json")

workspace, temp_dir = create_workspace(task)

result = run_verifier(task, workspace)

print(result)
