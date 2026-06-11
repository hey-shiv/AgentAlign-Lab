import json

from pydantic import ValidationError

from agentalign.schemas import Action


def parse_action(raw_text: str) -> Action | tuple[None, str]:
    """
    Parses an action from raw LLM output.
    Returns (Action, None) on success, or (None, error_message) on failure.
    """
    # Extract JSON. Look for first { and last }
    start = raw_text.find('{')
    end = raw_text.rfind('}')

    if start == -1 or end == -1 or start > end:
        return None, "Error: No JSON object found in output."

    json_str = raw_text[start:end+1]

    try:
        data = json.loads(json_str)
        action_name = data.get("action")

        if not action_name:
            return None, "Error: JSON missing 'action' field."

        return Action(name=action_name, args=data.get("args", {})), None
    except json.JSONDecodeError as e:
        return None, f"Error: Invalid JSON ({str(e)})."
    except ValidationError as e:
        return None, f"Error: Schema validation failed ({str(e)})."
    except Exception as e:
        return None, f"Error: Unexpected parsing failure ({str(e)})."


class OutputParser:
    """Compatibility parser for simple text-output tests."""

    @staticmethod
    def parse_action(output: str) -> str:
        return output.strip().splitlines()[0] if output.strip() else ""

    @staticmethod
    def is_valid_output(output: str) -> bool:
        return bool(output.strip())

    @staticmethod
    def parse_reasoning(output: str) -> str:
        return output.strip()
