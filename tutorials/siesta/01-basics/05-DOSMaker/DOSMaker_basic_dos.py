#!/usr/bin/env python
"""
Tutorial: Basic DOS (Density of States) Calculation

This tutorial demonstrates how to calculate the total electronic density of states
using DOSMaker. The DOS shows the distribution of electronic states as a function
of energy.

Use Case:
---------
When you want to analyze the electronic structure and determine:
- Band gap (if present)
- Fermi level position
- Distribution of valence and conduction states
- Metallicity of the system

Example:
--------
Silicon with total DOS calculation.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import DOSMaker
from jobflow import run_locally

# Load silicon structure
structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

print("\n" + "=" * 70)
print("Tutorial: Basic DOS Calculation")
print("=" * 70)
print(f"\nStructure: {structure.composition}")
print(f"Number of atoms: {len(structure)}")

# Create DOS maker with default settings
dos_maker = DOSMaker.dos_calculation(
    dry_run=True,
    user_params={
        "a2s_kpts": [8, 8, 8],  # SCF k-points
        "Mesh.Cutoff": "300 Ry",
        "%block ProjectedDensityOfStates": ["EF -1.000 1.000 0.0100 201 eV"],
    },
)


job = dos_maker.make(structure)
response = run_locally(job, create_folders=True)
