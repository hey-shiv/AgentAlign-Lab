#!/usr/bin/env python
"""Launch the dashboard.

This script launches the Streamlit dashboard for result visualization.
"""

import subprocess
import sys


def main():
    """Main entry point."""
    dashboard_path = "src/agentalign/dashboard/app.py"

    print(f"Launching dashboard from {dashboard_path}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])


if __name__ == "__main__":
    main()
