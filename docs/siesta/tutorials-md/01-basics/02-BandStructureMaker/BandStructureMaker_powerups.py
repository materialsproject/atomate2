#!/usr/bin/env python
"""Band structure showing multiple powerups modifications."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import BandStructureMaker
from atomate2.siesta.powerups import update_user_siesta_settings

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

# Create maker
maker = BandStructureMaker(dry_run=True)
job = maker.make(structure)

# First powerup: basic settings
job = update_user_siesta_settings(
    job,
    {
        "a2s_kpts": [6, 6, 6],
        "PAO.BasisSize": "DZ",
        "Mesh.Cutoff": "250 Ry",
    },
)

# Second powerup: SCF convergence settings
job = update_user_siesta_settings(
    job,
    {
        "SCF.Mixer.Weight": "0.05",
        "SCF.Mixer.History": "12",
        "OccupationFunction": "MP",
        "OccupationMPOrder": "2",
    },
)

results = run_locally(job, create_folders=True, root_dir="band_structure_powerups")

print("✓ Complete. Check: band_structure_powerups/")
print(
    "Verify: grep 'Mesh.Cutoff\\|SCF.Mixer\\|OccupationFunction' band_structure_powerups/job_*/siesta.fdf"
)
