"""Task generation utilities.

Provides functionality for creating and generating new tasks.
"""

from agentalign.schemas import Task


def generate_tasks(num_tasks: int, template: str = "default") -> list[Task]:
    """Generate a batch of tasks.

    Args:
        num_tasks: Number of tasks to generate
        template: Task template to use

    Returns:
        List of generated tasks
    """
    tasks = []
    for i in range(num_tasks):
        task = Task(
            task_id=f"task_{i:04d}",
            description=f"Task {i} using {template} template",
            metadata={"template": template, "index": i},
        )
        tasks.append(task)
    return tasks
