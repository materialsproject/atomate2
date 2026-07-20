#!/usr/bin/env python
"""
Optical Properties Calculation - Direct SIESTA FDF format

Shows how to enable optical properties calculation using direct SIESTA syntax.
SIESTA calculates dielectric function, absorption, refraction, etc.
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

# Silicon structure
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Direct FDF format for optical properties
user_params = {
    "xc.functional": "GGA",
    "xc.authors": "PBE",
    "Mesh.Cutoff": "300 Ry",
    "a2s_kpts": [8, 8, 8],  # Dense k-grid needed for optical properties
    "PAO.BasisSize": "DZP",
    # Optical calculation parameters
    "OpticalCalculation": "true",
    "Optical.Energy.Minimum": "0.0 eV",
    "Optical.Energy.Maximum": "10.0 eV",
    "Optical.Broaden": "0.1 eV",
    "Optical.Scissor": "0.0 eV",
}

# Create maker with dry_run
maker = StaticMaker.scf(
    user_params=user_params, dry_run=True, dry_run_output_dir="optical_preview"
)

job = maker.make(structure)

# Run with dry_run
results = run_locally(job, create_folders=True)

print(f"\nDry-run completed: {results}")
print("Check optical_preview/ for generated SIESTA input files")
print("Output files (when run): siesta.EPSIMG, siesta.EPSREAL (dielectric function)")
