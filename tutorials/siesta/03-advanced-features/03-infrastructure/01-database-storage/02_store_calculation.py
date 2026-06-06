#!/usr/bin/env python
"""Store calculation results in MongoDB."""

from pathlib import Path
from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

# Check for config
config_file = Path.home() / ".jobflow.yaml"
if not config_file.exists():
    print("✗ Error: ~/.jobflow.yaml not found")
    print("  Run: atomate2siesta-database config --generate")
    exit(1)

print("✓ Config found")

# Dry-run first to validate
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True, dry_run_output_dir="db_preview")
job = maker.make(structure)
run_locally(job, create_folders=True)

print("✓ Dry-run complete: db_preview/")
print("  Check siesta.fdf, then remove dry_run=True to store in MongoDB")
