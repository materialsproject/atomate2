#!/usr/bin/env python3
"""
Tighten Convergence Criteria
=============================

Make convergence tighter than preset defaults.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

structure = Structure.from_file("../../../../00-structures/Si_mp-149_primitive.cif")

# Tighten convergence
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(
    maker,
    "relax_high_accuracy",
    override_params={
        "SCF.DM.Tolerance": 1e-7,
        "MD.MaxForceTol": "0.001 eV/Ang",
        "SCF.Mixer.Weight": 0.02,
    },
)

job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✅ Ultra-tight convergence for critical calculations")
