#!/usr/bin/env python3
"""Script 07: Launch the Gradio dashboard.

Usage:
    python scripts/07_launch_dashboard.py
"""

import socket


def _find_free_port(start: int = 7860, end: int = 7880) -> int:
    """Find a free port in the given range."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    return start  # fallback


def main() -> None:
    from agentalign.dashboard.app import build_app

    port = _find_free_port()
    print(f"Launching AgentAlign Dashboard on http://localhost:{port}")
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
