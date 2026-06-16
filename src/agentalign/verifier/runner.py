"""Verifier runner — delegates to checks.run_verifier.

This module exists for backward compatibility. The main verifier logic
lives in checks.py.
"""

from agentalign.verifier.checks import run_verifier  # noqa: F401
