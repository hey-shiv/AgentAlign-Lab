"""Agent output parsing.

Utilities for parsing and extracting structured information from agent outputs.
"""


class OutputParser:
    """Parser for agent outputs."""

    @staticmethod
    def parse_action(output: str) -> str:
        """Extract action from agent output.

        Args:
            output: Raw agent output

        Returns:
            Parsed action
        """
        # Placeholder implementation
        lines = output.strip().split("\n")
        return lines[0] if lines else "unknown"

    @staticmethod
    def parse_reasoning(output: str) -> str:
        """Extract reasoning from agent output.

        Args:
            output: Raw agent output

        Returns:
            Extracted reasoning
        """
        # Placeholder implementation
        return output.strip()

    @staticmethod
    def is_valid_output(output: str) -> bool:
        """Check if output is valid.

        Args:
            output: Agent output to validate

        Returns:
            True if valid
        """
        return len(output.strip()) > 0
