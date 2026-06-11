import argparse
import os
import socket
import sys


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    port = args.port or find_free_port()

    print("Launching AgentAlign Lab Dashboard...")
    print(f"Dashboard URL: http://127.0.0.1:{port}", flush=True)
    # Add src to pythonpath if not running through a proper module
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from agentalign.dashboard.app import build_app

    app = build_app()
    app.launch(share=False, server_port=port)


if __name__ == "__main__":
    main()
