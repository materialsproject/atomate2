#!/usr/bin/env python
"""Relaxation with explicit DM.InitSpin block specification."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

# Method 2: Explicitly specify DM.InitSpin block via fdf_arguments
# This gives you full control over the initial spin configuration
maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=False,
    user_params={
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [2, 2, 2],
        "xc.functional": "GGA",
        "xc.authors": "PBE",
        "a2s_pseudo_relativistic": "SR",
        "Spin": "polarized",
        # Explicitly define the DM.InitSpin block
        # Format: atom_index magnetic_moment
        "%block DM.InitSpin": [
            "1  2.0",  # First Si atom: 2.0 μB
            "2 -1.0",  # Second Si atom: -2.0 μB (antiferromagnetic)
        ],
    },
)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ Relax with explicit DM.InitSpin complete")
print("  Generated siesta.fdf includes:")
print("    - Spin polarized")
print("    - DM.InitSpin block with antiferromagnetic configuration")
print("    - First atom: +2.0 μB")
print("    - Second atom: -2.0 μB")
