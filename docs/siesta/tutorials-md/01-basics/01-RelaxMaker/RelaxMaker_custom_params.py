#!/usr/bin/env python
"""Relaxation with custom parameters."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")


maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    force_unknown=True,
    user_params={
        # "PAO.BasisSize": "DZP",
        # "kgrid.Cutoff": "10 Ang",
        # "a2s_kpts": [2, 2, 2],  # Internal parameter (recommended)
        # Alternative: Use FDF block format (advanced)
        "%block kgrid.monkhorst.pack": [
            [2, 0, 0, 0.0],
            [0, 2, 0, 0.0],
            [0, 0, 2, 0.0],
        ],
        "xc.functional": "GGA",
        "xc.authors": "PBE",
        "a2s_pseudo_relativistic": "SR",
        "mesh.cutoff": "500 Ry",
        "new": "test",
    },
)

job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ Relax with custom parameters complete")
print("  Generated siesta.fdf includes:")
