#!/usr/bin/env python3
"""
Add DFT+U Corrections
=====================

Add DFT+U for correlated electron systems.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

structure = Structure.from_file("../../../../00-structures/Si_mp-149_primitive.cif")

# Add DFT+U
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(
    maker,
    "relax_standard",
    override_params={
        "Spin": "polarized",
        "DFTU.ProjectorGenerationMethod": 2,
        "DFTU.CutoffNorm": 0.9,
        "%block DFTU.Proj": [
            "Si 1                 # label, number of l-shells with U",
            "n=3 1                # n=3 (3p), l=1 (p-shell)",
            "5.0 0.0              # U (eV), J (eV)",
            "0.0 0.0              # rc, omega",
        ],
        "a2s_kpts": [6, 6, 6],
    },
)

job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✅ Added DFT+U to standard preset")
print("   For correlated electron systems")
