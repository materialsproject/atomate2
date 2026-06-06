#!/usr/bin/env python3
"""
Add Spin Polarization
======================

Add magnetic parameters to a preset.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

structure = Structure.from_file("../../../../00-structures/Si_mp-149_primitive.cif")

# Add spin polarization
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(
    maker,
    "relax_standard",
    override_params={
        "Spin": "polarized",
        "%block DM.InitSpin": [
            "1  2.0",
            "2  2.0",
        ],
        "SCF.Mixer.Weight": 0.002,
        "SCF.Mixer.History": 10,
    },
)

job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✅ Added spin polarization to standard preset")
print("   Check siesta.fdf for DM.InitSpin block")
