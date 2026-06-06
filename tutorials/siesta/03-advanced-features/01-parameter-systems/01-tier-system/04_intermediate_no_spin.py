#!/usr/bin/env python
"""Intermediate tier without spin (for faster calculations)."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

maker = RelaxMaker.fixed_cell_relaxation(
    tier="intermediate", disabled_modules=["spin"], dry_run=True
)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ Intermediate tier without spin complete")
