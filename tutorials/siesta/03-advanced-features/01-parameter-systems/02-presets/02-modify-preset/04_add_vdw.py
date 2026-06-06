#!/usr/bin/env python3
"""
Add van der Waals Corrections
==============================

Add vdW-DF functional to a preset.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

structure = Structure.from_file("../../../../00-structures/Si_mp-149_primitive.cif")

# Add vdW corrections
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(
    maker,
    "2d_semiconductor",
    override_params={
        "XC.functional": "VDW",
        "XC.authors": "DRSLL",
        "Mesh.Cutoff": "350 Ry",
    },
)

job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✅ Added vdW corrections to 2D preset")
print("   Perfect for TMD bilayers")
