#!/usr/bin/env python3
"""
Add Geometry Constraints
=========================

Add fixed atoms for surface calculations.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

structure = Structure.from_file("../../../../00-structures/Si_mp-149_primitive.cif")

# Add constraints
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(
    maker,
    "surface_metal",
    override_params={
        "%block Geometry.Constraints": ["position from 1 to 3"],
        "MD.MaxForceTol": "0.02 eV/Ang",
    },
)

job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✅ Added constraints to surface preset")
print("   Useful for fixed substrate layers")
