#!/usr/bin/env python
"""Variable-cell relaxation (one-liner pattern)."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

job = RelaxMaker.variable_cell_relaxation().make(structure)
results = run_locally(job, create_folders=True)

print("✓ Variable-cell relaxation complete")
