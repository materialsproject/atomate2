#!/usr/bin/env python
"""Basic band structure calculation."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import BandStructureMaker

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

job = BandStructureMaker.bandstructure_calculation(tier="dirty", dry_run=True).make(
    structure
)
results = run_locally(job, create_folders=True, root_dir="01_basic")

print("✓ Band structure complete")
