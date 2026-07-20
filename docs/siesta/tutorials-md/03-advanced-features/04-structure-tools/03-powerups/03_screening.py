#!/usr/bin/env python
"""High-throughput screening with powerups."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.powerups import update_user_siesta_settings, add_metadata

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

basis_sets = ["SZ", "DZ", "DZP", "DZDP"]

print(f"Screening {len(basis_sets)} basis sets:\n")

for basis in basis_sets:
    maker = StaticMaker(dry_run=True, dry_run_output_dir=f"screening_{basis}")
    job = maker.make(structure)

    # Apply basis-specific parameters
    job = update_user_siesta_settings(job, {"PAO.BasisSize": basis})
    job = add_metadata(job, {"basis_test": basis})

    run_locally(job, create_folders=True)
    print(f"  ✓ {basis:5s}: screening_{basis}/")

print("\n✓ Screening complete")
