#!/usr/bin/env python3
"""
Modify Multiple Parameters
===========================

Override multiple parameters at once.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

structure = Structure.from_file("../../../../00-structures/Si_mp-149_primitive.cif")

# Modify multiple parameters
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(
    maker,
    "adsorbate_screening",
    override_params={
        "a2s_kpts": [6, 6, 1],
        "Mesh.Cutoff": "250 Ry",
        "SCF.Mixer.Weight": 0.005,
    },
)

job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✅ Modified 3 parameters for more accurate screening")
