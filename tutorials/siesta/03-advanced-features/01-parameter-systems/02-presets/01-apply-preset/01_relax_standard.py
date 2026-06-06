#!/usr/bin/env python3
"""
Apply Standard Relaxation Preset
=================================

Learn how to apply the relax_standard preset to a structure.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

# Load structure
structure = Structure.from_file("../../../../00-structures/Si_mp-149_primitive.cif")

# Create maker with dry_run and apply preset
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(maker, "relax_standard")

# Run job
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✅ Preset applied: relax_standard (Tier: intermediate)")
print("   Check: job_*/dry_run_output/*/siesta.fdf")
