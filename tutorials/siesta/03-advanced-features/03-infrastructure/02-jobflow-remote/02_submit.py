#!/usr/bin/env python
"""Submit job to jobflow-remote local worker.

This tutorial demonstrates actual job submission to jobflow-remote,
even for a local worker. This is useful for:
- Testing jobflow-remote infrastructure
- Running jobs asynchronously
- Queue management for local jobs
- Database tracking
"""

from pathlib import Path
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

# Check prerequisites
config_file = Path.home() / ".jfremote" / "atomate2siesta.yaml"
if not config_file.exists():
    print("✗ Config not found")
    print("  Run: atomate2siesta-jobflow-remote setup")
    exit(1)

print("✓ Config found")

# Check if jobflow-remote is installed
try:
    from jobflow_remote import submit_flow

    print("✓ jobflow-remote installed")
except ImportError:
    print("✗ jobflow-remote not installed")
    print("  Run: atomate2siesta-jobflow-remote install")
    exit(1)

# Create a simple relaxation job
print("\n→ Creating relaxation job...")
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")
maker = RelaxMaker.fixed_cell_relaxation()
job = maker.make(structure)
print("✓ Job created")

# Submit to jobflow-remote
print("\n→ Submitting to jobflow-remote...")
print("  Worker: local_shell (from config)")
print("  Project: atomate2siesta")

try:
    job_id = submit_flow(job, worker="local_shell", project="atomate2siesta")
    print(f"✓ Job submitted: {job_id}")

    print("\n→ Monitor job:")
    print(f"  jf -p atomate2siesta job info {job_id}")
    print("  jf -p atomate2siesta job list")
    print("  jf -p atomate2siesta runner check")

    print("\n→ Next steps:")
    print("  1. Start runner if not running: jf -p atomate2siesta runner start")
    print("  2. Check job status: jf -p atomate2siesta job list")
    print("  3. View output: jf -p atomate2siesta job info <job_id>")

except Exception as e:
    print(f"✗ Submission failed: {e}")
    print("\n→ Troubleshooting:")
    print("  1. Check runner is started: jf -p atomate2siesta runner check")
    print("  2. Check MongoDB is running: jf -p atomate2siesta admin check")
    print("  3. View logs: jf -p atomate2siesta runner logs")
    print("\n  To start runner: jf -p atomate2siesta runner start")
