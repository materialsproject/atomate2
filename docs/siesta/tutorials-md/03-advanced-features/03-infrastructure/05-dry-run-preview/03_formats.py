#!/usr/bin/env python
"""Test different output formats for dry-run."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import StaticMaker

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

formats = ["fdf", "XV", "cif", "xsf", "json", "POSCAR"]

for fmt in formats:
    maker = StaticMaker(
        dry_run=True, dry_run_output_dir=f"format_{fmt}", dry_run_format=fmt
    )
    job = maker.make(structure)
    run_locally(job, create_folders=True)
    print(f"✓ {fmt.upper():7s}: format_{fmt}/")
