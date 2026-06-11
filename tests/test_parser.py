"""Tests for parser module."""

from agentalign.agent.parser import OutputParser


def test_parse_action():
    """Test parsing action from output."""
    output = "Search Google for information\nOther output"
    action = OutputParser.parse_action(output)
    assert action == "Search Google for information"


def test_is_valid_output():
    """Test output validation."""
    assert OutputParser.is_valid_output("some output") is True
    assert OutputParser.is_valid_output("") is False
    assert OutputParser.is_valid_output("   ") is False


def test_parse_reasoning():
    """Test extracting reasoning."""
    output = "Let me think about this"
    reasoning = OutputParser.parse_reasoning(output)
    assert "think" in reasoning
