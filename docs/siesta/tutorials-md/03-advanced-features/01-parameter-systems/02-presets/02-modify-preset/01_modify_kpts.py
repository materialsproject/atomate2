#!/usr/bin/env python3
"""
Modify k-points from Preset
============================

Override a single parameter from the preset.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

structure = Structure.from_file("../../../../00-structures/Si_mp-149_primitive.cif")

# Modify kpts: [4,4,4] → [6,6,6]
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(
    maker,
    "relax_standard",
    override_params={"a2s_kpts": [6, 6, 6]},
)

job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✅ Modified: kpts [4,4,4] → [6,6,6]")
print("   Other preset params unchanged")
