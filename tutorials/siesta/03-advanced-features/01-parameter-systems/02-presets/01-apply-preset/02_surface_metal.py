#!/usr/bin/env python3
"""
Apply Surface Metal Preset
===========================

Surface metal preset includes MP smearing and slow SCF mixing.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

structure = Structure.from_file("../../../../00-structures/Si_mp-149_primitive.cif")

maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(maker, "surface_metal")

job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✅ Preset applied: surface_metal (Tier: intermediate)")
print("   Includes: MP smearing, kpts=[6,6,1], slow SCF mixing")
