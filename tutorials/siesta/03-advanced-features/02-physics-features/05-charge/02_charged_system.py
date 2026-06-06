#!/usr/bin/env python
"""
Charged System Calculation - Direct SIESTA FDF format

Shows how to simulate charged systems (ions, charged defects, etc.).
SIESTA adds compensating background charge for neutrality.
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

# Silicon with vacancy (charged defect)
lattice = Lattice.cubic(10.86)
structure = Structure(
    lattice,
    ["Si"] * 7,  # 8-atom cell with 1 vacancy
    [
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
    ],
)

# Direct FDF format for charged system
user_params = {
    "XC.functional": "GGA",  # SIESTA FDF parameter
    "Mesh.Cutoff": "300 Ry",  # SIESTA FDF parameter
    "a2s_kpts": [2, 2, 2],  # Internal parameter for k-points
    "PAO.BasisSize": "DZP",
    # Net charge (electrons removed/added)
    # Positive = electrons removed, Negative = electrons added
    # Accepts: 2.0, "+2.0", "-2.0", etc.
    "NetCharge": "+2.0",  # +2 charge (2 electrons removed, e.g., V²⁺)
}

# Create maker with dry_run
maker = StaticMaker.scf(
    user_params=user_params, dry_run=True, dry_run_output_dir="charged_preview"
)

job = maker.make(structure)

# Run with dry_run
results = run_locally(job, create_folders=True)

print(f"\nDry-run completed: {results}")
print("Check charged_preview/ for generated SIESTA input files")
print("System has net charge: +2.0 (2 electrons removed)")
