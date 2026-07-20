#!/usr/bin/env python
"""
Tutorial: PDOS (Projected Density of States) for All Atoms

This tutorial demonstrates how to calculate projected density of states (PDOS).
PDOS provides orbital-resolved contributions from all atoms in your structure.

Use Case:
---------
When you want to analyze:
- Orbital contributions from each atom (s, p, d, f orbitals)
- Bonding character and hybridization
- Element-specific contributions to DOS
- d-orbital splitting in transition metals
- Band character analysis

Example:
--------
MgO with PDOS for all atoms (both Mg and O).
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import PDOSMaker
from jobflow import run_locally

# Load MgO structure
structure = Structure.from_file("../../00-structures/MgO_mp-1265_primitive.cif")

print("\n" + "=" * 70)
print("Tutorial: PDOS for Specific Atoms")
print("=" * 70)
print(f"\nStructure: {structure.composition}")
print(f"Number of atoms: {len(structure)}")

# Print atom information
print("\nAtom information:")
for i, site in enumerate(structure, start=1):
    print(f"  Atom {i}: {site.species_string} at {site.frac_coords}")

# Create PDOS maker - SIESTA will automatically generate PDOS for ALL atoms
pdos_maker = PDOSMaker.pdos_calculation(
    dry_run=True,
    user_params={
        "a2s_kpts": [8, 8, 8],  # SCF k-points
        "Mesh.Cutoff": "350 Ry",
        # "%block ProjectedDensityOfStates": [ "Ef -5.0 2.0 0.05 300 eV"]
        "%block ProjectedDensityOfStates": ["-5.0 2.0 0.05 300 eV"],
        "%block PDOS.kgrid.MonkhorstPack": [
            "15 0 0 0.0",
            "0 15 0 0.0",
            "0 0 15 0.0",
        ],
    },
)


job = pdos_maker.make(structure)
response = run_locally(job, create_folders=True)

print("\n✓ Generated FDF with PDOS settings")
