from agentalign.loader import load_task
from agentalign.workspace import create_workspace

task = load_task("data/tasks/py_fix_001.json")

workspace, temp_dir = create_workspace(task)

print("Workspace: ", workspace)

print("\nFiles created:")

for path in workspace.iterdir():
    print("_", path.name)

print("\nContents of stats.py:")

print((workspace/ "stats.py").read_text())