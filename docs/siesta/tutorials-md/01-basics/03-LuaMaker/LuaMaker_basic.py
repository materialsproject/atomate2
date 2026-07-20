#!/usr/bin/env python
"""Basic FLOS/Lua script usage."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import LuaMaker

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

maker = LuaMaker.fixed_cell_relaxation(dry_run=True)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ FLOS/Lua relax complete")
