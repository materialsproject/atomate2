#!/usr/bin/env python
"""Explore jobflow-remote CLI setup."""

import subprocess

commands = [
    ("--help", "atomate2siesta-jobflow-remote --help", True),
    ("info", "atomate2siesta-jobflow-remote info", False),
    ("runner", "atomate2siesta-jobflow-remote runner", False),
]

print("Jobflow-remote CLI commands:\n")

for desc, cmd, show_output in commands:
    print(f"→ {desc}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if show_output:
        # Show first few lines for help text
        lines = result.stdout.split("\n")[:10]
        print("\n".join(lines))
    else:
        # Just show success indicator for commands with wide Rich output
        if result.returncode == 0:
            print("  ✓ Command available")
        else:
            print(f"  ⚠ Error: {result.stderr[:50]}")
    print()

print("✓ Setup workflow:")
print("  1. atomate2siesta-jobflow-remote install")
print("  2. atomate2siesta-jobflow-remote setup")
print("  3. jf -p atomate2siesta admin reset")
print("  4. jf -p atomate2siesta runner start")
