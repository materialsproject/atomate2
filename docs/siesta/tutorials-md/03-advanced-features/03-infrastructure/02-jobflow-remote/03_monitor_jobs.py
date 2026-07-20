#!/usr/bin/env python
"""Monitor jobflow-remote jobs."""

import subprocess

PROJECT = "atomate2siesta"

commands = [
    f"jf -p {PROJECT} job list",
    f"jf -p {PROJECT} runner status",
]

print("Job monitoring:\n")

for cmd in commands:
    print(f"→ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout if result.stdout else "  ✓ Command executed successfully")
    else:
        print(
            f"  ⚠ Error: {result.stderr[:100] if result.stderr else 'Command failed'}"
        )
    print()

print("✓ Monitoring commands:")
print(f"  jf -p {PROJECT} job info <job_id>")
print(f"  jf -p {PROJECT} job output <job_id>")
print(f"  jf -p {PROJECT} job rerun <job_id>")
