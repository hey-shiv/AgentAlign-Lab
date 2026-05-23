"""Agent prompts and prompt templates.

Defines system prompts, few-shot examples, and prompt templates.
"""


class PromptTemplate:
    """Template for agent prompts."""

    SYSTEM_PROMPT = """You are a helpful AI agent. 
Solve the given task step by step.
Be clear about your reasoning."""

    @staticmethod
    def format_task(task_description: str) -> str:
        """Format a task for the agent.

        Args:
            task_description: Description of the task

        Returns:
            Formatted prompt
        """
        return f"Task: {task_description}\n\nPlease solve this step by step."

    @staticmethod
    def format_history(steps: list[dict]) -> str:
        """Format trajectory history.

        Args:
            steps: List of steps taken

        Returns:
            Formatted history
        """
        history = []
        for step in steps:
            history.append(f"Step {step['id']}: {step['action']}")
            history.append(f"Observation: {step['observation']}")
        return "\n".join(history)
