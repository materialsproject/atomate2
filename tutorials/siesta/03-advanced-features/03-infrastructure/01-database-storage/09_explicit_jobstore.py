#!/usr/bin/env python
"""Explicitly define JobStore without jobflow.yaml."""

from pathlib import Path
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import SETTINGS, JobStore, run_locally
from maggma.stores import MongoStore
from pymatgen.core import Structure

# Define MongoDB store
store = MongoStore(
    database="atomate2siesta",
    collection_name="tasks",
    host="localhost",
    port=27017,
)

# Create JobStore and set in SETTINGS
job_store = JobStore(docs_store=store)
SETTINGS.JOB_STORE = job_store

# Create job
structure_file = (
    Path(__file__).parent.parent.parent.parent
    / "00-structures"
    / "Si_mp-149_primitive.cif"
)
structure = Structure.from_file(structure_file)
maker = RelaxMaker.fixed_cell_relaxation()
job = maker.make(structure)

# Run calculation (uses SETTINGS.JOB_STORE set above)
# results = run_locally(job, create_folders=True)

# Dry-run example
maker_dry = RelaxMaker.fixed_cell_relaxation(
    dry_run=False, dry_run_output_dir="explicit_jobstore_preview"
)
job_dry = maker_dry.make(structure)
run_locally(job_dry, create_folders=True)

print("✓ Dry-run complete")
print("  Uncomment line 31 to run actual calculation with explicit JobStore")
