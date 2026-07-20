#!/usr/bin/env python
"""Relaxation with custom parameters."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

# Method 1: Set magnetic moments on the structure
# This will automatically write the DM.InitSpin block
structure.add_site_property("magmom", [1.0, 1.0])  # 1.0 μB per Si atom

maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=False,
    user_params={
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [2, 2, 2],
        "xc.functional": "GGA",
        "xc.authors": "PBE",
        "a2s_pseudo_relativistic": "SR",
        "Spin": "polarized",
    },
)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ Relax with custom parameters complete")
print("  Generated siesta.fdf includes:")
print("    - Spin polarized")
print("    - DM.InitSpin block with initial magnetic moments")
