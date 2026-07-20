#!/usr/bin/env python
"""Basic tier (6 modules) - minimal parameter set."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

maker = RelaxMaker.fixed_cell_relaxation(tier="basic", dry_run=True)
# maker = RelaxMaker.fixed_cell_relaxation(tier="intermediate", dry_run=True)
# maker = RelaxMaker.fixed_cell_relaxation(tier="advanced", dry_run=True)
# maker = RelaxMaker.fixed_cell_relaxation(tier="expert", dry_run=True)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ Basic tier relaxation complete")
