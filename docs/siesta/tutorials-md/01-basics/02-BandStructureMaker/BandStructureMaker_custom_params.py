#!/usr/bin/env python
"""Band structure with custom user_params."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import BandStructureMaker

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

# Create maker with custom parameters
maker = BandStructureMaker.bandstructure_calculation(
    dry_run=True,
    user_params={
        "a2s_kpts": [8, 8, 8],
        "PAO.BasisSize": "DZP",
        "Mesh.Cutoff": "300 Ry",
        "SCF.DM.Tolerance": "1.0e-5",
        "ElectronicTemperature": "25 meV",
    },
)

job = maker.make(structure)
results = run_locally(job, create_folders=True, root_dir="band_structure_custom")

print("✓ Complete. Check: band_structure_custom/")
print(
    "Verify: grep 'kgrid\\|PAO.BasisSize\\|Mesh.Cutoff' band_structure_custom/job_*/siesta.fdf"
)
