#!/usr/bin/env python
"""Explore atomate2siesta-database CLI commands."""

import subprocess

commands = [
    ("--help", "Show all commands", False),
    ("config --generate", "Generate ~/.jobflow.yaml (interactive - skipped)", True),
    ("test", "Test MongoDB connection", False),
    ("stats", "Show database statistics", False),
    ("list --limit 5", "List recent documents", False),
]

print("Exploring database CLI commands:\n")

for cmd, desc, skip_interactive in commands:
    full_cmd = f"atomate2siesta-database {cmd}"
    print(f"→ {desc}")

    if skip_interactive:
        # print(f"  ⚠ Skipped (requires user input)")
        # print(f"  Run manually: {full_cmd}")
        # print()
        continue

    print(f"  Running: {full_cmd}")
    result = subprocess.run(
        full_cmd, shell=True, capture_output=True, text=True, timeout=5
    )
    if "not found" in result.stderr.lower() or result.returncode != 0:
        if "test" in cmd:
            print("  ⚠ MongoDB not running (expected)")
        else:
            print("  ⚠ Command failed")
    # else:
    #    print("  ✓")
    print()  # Blank line between commands


print("✓ CLI exploration complete")
print("  Run: atomate2siesta-database config --generate")
print("  Then: atomate2siesta-database test")
