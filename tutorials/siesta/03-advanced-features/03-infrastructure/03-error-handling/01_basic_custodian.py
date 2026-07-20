#!/usr/bin/env python
"""Test custodian on difficult SCF convergence."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.powerups import update_user_siesta_settings

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=False, use_custodian=True, custodian_max_errors=15
)
job = maker.make(structure)

# Make SCF harder to test error recovery
job = update_user_siesta_settings(job, {"SCF.Mixer.Weight": 0.5, "MaxSCFIterations": 5})

results = run_locally(job, create_folders=True)
print("✓ Difficult SCF test complete")
print("  Custodian should have adjusted mixer parameters")
