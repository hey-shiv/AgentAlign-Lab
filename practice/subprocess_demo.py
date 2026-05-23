import subprocess

result = subprocess.run(
    ["echo","hello world"],
    capture_output=True,
    text=True
)

print(result)
print(result.stdout)
print(result.returncode)
