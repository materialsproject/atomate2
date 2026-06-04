#!/usr/bin/env python3
"""
Apply 2D Metal Preset
======================

Dense in-plane k-points for 2D materials.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

structure = Structure.from_file("../../../../00-structures/Si_mp-149_primitive.cif")

maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(maker, "2d_metal")

job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✅ Preset applied: 2d_metal (Tier: intermediate)")
print("   Dense in-plane kpts=[12,12,1] for 2D materials")
