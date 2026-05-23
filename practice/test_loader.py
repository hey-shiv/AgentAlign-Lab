from agentalign.loader import load_task


task = load_task("data/tasks/py_fix_001.json")

print(task)

print(type(task))
print(type(task.files[0]))
print(type(task.verifier))