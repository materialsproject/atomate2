#!/usr/bin/env python
"""
Basic Phonon Input Parameters - Direct SIESTA FDF format

Shows how to set phonon-related parameters using direct SIESTA syntax.
These parameters control force constants calculation for phonon workflows.
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

# Silicon structure - use primitive cell (2 atoms, not 8-atom conventional)
structure_conv = Structure.from_spacegroup(
    "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]
)
structure = structure_conv.get_primitive_structure()

# Direct FDF format for phonon parameters
user_params = {
    "xc.functional": "GGA",
    "xc.authors": "PBE",
    "Mesh.Cutoff": "300 Ry",
    "a2s_kpts": [6, 6, 6],
    "PAO.BasisSize": "DZP",
    # Phonon-related FDF parameters
    "MD.TypeOfRun": "FC",  # Force constants calculation
    "MD.FCDispl": "0.04 Bohr",  # Displacement for force constants
    "MD.FCfirst": 1,  # First atom to displace
    "MD.FClast": 2,  # Last atom to displace
}

# Create maker with dry_run
maker = StaticMaker.scf(
    user_params=user_params, dry_run=True, dry_run_output_dir="phonon_input_preview"
)

job = maker.make(structure)

# Run with dry_run
results = run_locally(job, create_folders=True)

print(f"\nDry-run completed: {results}")
print("Check phonon_input_preview/ for generated SIESTA input files")
print("FDF parameters: MD.TypeOfRun, MD.FCDispl, MD.FCfirst, MD.FClast")
