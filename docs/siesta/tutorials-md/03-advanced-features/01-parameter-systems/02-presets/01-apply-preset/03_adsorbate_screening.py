#!/usr/bin/env python3
"""
Apply Adsorbate Screening Preset
=================================

Fast preset for scanning many adsorption sites.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

structure = Structure.from_file("../../../../00-structures/Si_mp-149_primitive.cif")

maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(maker, "adsorbate_screening")

job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✅ Preset applied: adsorbate_screening (Tier: basic)")
print("   Fast for scanning 100+ adsorption sites")
