import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path


with TemporaryDirectory() as tmp:
    workspace = Path(tmp)

    (workspace / "hello.txt").write_text("sandbox file")

    result = subprocess.run(
        ["ls"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=5
    )

    print(result.stdout)