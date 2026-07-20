#!/usr/bin/env python
"""Surface metal preset - optimized for metallic surfaces."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(maker, "surface_metal")
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ Surface metal preset relaxation complete")
