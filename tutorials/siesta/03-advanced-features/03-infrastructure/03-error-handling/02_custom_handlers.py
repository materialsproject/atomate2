#!/usr/bin/env python
"""Custom error handlers with max errors."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True, use_custodian=True, custodian_max_errors=10
)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ Relax with custom error handling (max 10 corrections)")
