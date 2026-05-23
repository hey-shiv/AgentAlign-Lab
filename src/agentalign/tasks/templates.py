"""Task templates and template utilities.

Provides pre-defined task templates for common scenarios.
"""


class TaskTemplate:
    """Base class for task templates."""

    name: str = "base"
    description: str = "Base task template"

    @classmethod
    def render(cls, **kwargs) -> dict:
        """Render template with given parameters.

        Returns:
            Dictionary with rendered template
        """
        return {"template": cls.name, "params": kwargs}


class MathTemplate(TaskTemplate):
    """Template for math problems."""

    name = "math"
    description = "Mathematical problem template"


class LogicTemplate(TaskTemplate):
    """Template for logic puzzles."""

    name = "logic"
    description = "Logic puzzle template"


AVAILABLE_TEMPLATES = {
    "math": MathTemplate,
    "logic": LogicTemplate,
}
