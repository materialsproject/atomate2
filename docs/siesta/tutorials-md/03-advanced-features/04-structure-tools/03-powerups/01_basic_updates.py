#!/usr/bin/env python
"""Basic parameter updates with powerups."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.powerups import update_user_siesta_settings

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Create job
maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True, dry_run_output_dir="powerup_basic"
)
job = maker.make(structure)

# Apply powerup to update parameters
job = update_user_siesta_settings(
    job,
    {
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [6, 6, 6],
        "Mesh.Cutoff": "300 Ry",
        "SCF.Mixer.Weight": 0.05,
    },
)

run_locally(job, create_folders=True)
print("✓ Complete: powerup_basic/")
print("  Check siesta.fdf for updated parameters")
