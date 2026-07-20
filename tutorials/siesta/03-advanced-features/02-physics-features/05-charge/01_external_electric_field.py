#!/usr/bin/env python
"""
External Electric Field - Direct SIESTA FDF format

Shows how to apply external electric fields to systems.
Useful for piezoelectric materials, ferroelectrics, and field-effect devices.
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

# Silicon slab (for field effect study)
lattice = Lattice.cubic(10.0)
structure = Structure(
    lattice,
    ["Si", "Si", "Si", "Si"],
    [[0.25, 0.25, 0.3], [0.75, 0.75, 0.3], [0.25, 0.75, 0.7], [0.75, 0.25, 0.7]],
)

# Direct FDF format for external electric field
user_params = {
    "XC.functional": "GGA",  # SIESTA FDF parameter
    "Mesh.Cutoff": "300 Ry",  # SIESTA FDF parameter
    "a2s_kpts": [4, 4, 1],  # Internal parameter for k-points (2D slab)
    "PAO.BasisSize": "DZP",
    # External electric field in z-direction (0.01 V/Ang)
    # Format: Ex Ey Ez unit (V/Ang or Ry/Bohr)
    "%block ExternalElectricField": ["0.0  0.0  0.01  V/Ang"],
}

# Create maker with dry_run
maker = StaticMaker.scf(
    user_params=user_params, dry_run=True, dry_run_output_dir="efield_preview"
)

job = maker.make(structure)

# Run with dry_run
results = run_locally(job, create_folders=True)

print(f"\nDry-run completed: {results}")
print("Check efield_preview/ for generated SIESTA input files")
print("External electric field: 0.01 V/Ang in z-direction")
