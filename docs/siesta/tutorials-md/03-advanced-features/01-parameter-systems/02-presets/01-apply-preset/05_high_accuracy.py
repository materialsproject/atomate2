#!/usr/bin/env python3
"""
Apply High Accuracy Preset
===========================

TZP basis with tight convergence for high-quality results.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

structure = Structure.from_file("../../../../00-structures/Si_mp-149_primitive.cif")

maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(maker, "relax_high_accuracy")

job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✅ Preset applied: high_accuracy_relax (Tier: intermediate)")
print("   TZP basis, kpts=[8,8,8], tight convergence")
