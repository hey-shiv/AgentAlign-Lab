"""Agent tools and tool integration.

Defines tools available to the agent and manages tool execution.
"""

from typing import Any


class Tool:
    """Base class for agent tools."""

    name: str = "base_tool"
    description: str = "Base tool"

    def __call__(self, *args, **kwargs) -> Any:
        """Execute tool."""
        raise NotImplementedError


class CalculatorTool(Tool):
    """Simple calculator tool."""

    name = "calculator"
    description = "Perform arithmetic calculations"

    def __call__(self, expression: str) -> float:
        """Evaluate expression."""
        try:
            return float(eval(expression))  # Note: unsafe in production
        except Exception as e:
            return f"Error: {e}"


class SearchTool(Tool):
    """Placeholder search tool."""

    name = "search"
    description = "Search for information"

    def __call__(self, query: str) -> str:
        """Perform search."""
        return f"Search results for: {query}"


AVAILABLE_TOOLS = {
    "calculator": CalculatorTool(),
    "search": SearchTool(),
}
