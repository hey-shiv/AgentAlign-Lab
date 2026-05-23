from tempfile import TemporaryDirectory
from pathlib import Path


with TemporaryDirectory() as tmp:
    workspace = Path(tmp)

    file = workspace / "hello.txt"

    file.write_text("sandbox works")

    print(file.read_text())

    print(workspace)