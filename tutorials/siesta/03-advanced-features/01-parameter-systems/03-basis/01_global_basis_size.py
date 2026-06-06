#!/usr/bin/env python
"""
Tutorial: Global Basis Size Setting (PAO.BasisSize)

This tutorial demonstrates the simplest way to set basis size - using a global
PAO.BasisSize setting that applies to all atoms.

Available Options:
------------------
- SZ:  Single-Zeta (minimal basis, fastest)
- DZ:  Double-Zeta (better accuracy)
- SZP: Single-Zeta + Polarization
- DZP: Double-Zeta + Polarization (recommended default)
- TZP: Triple-Zeta + Polarization (high accuracy, slow)
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

# Load silicon structure
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

print("\n" + "=" * 70)
print("Tutorial: Global Basis Size (PAO.BasisSize)")
print("=" * 70)
print(f"\nStructure: {structure.composition}")
print(f"Number of atoms: {len(structure)}")

# Set global basis size to DZP (recommended for production)
user_params = {
    "PAO.BasisSize": "DZ",  # All atoms get Double-Zeta Polarized
    "a2s_kpts": [6, 6, 6],
    "Mesh.Cutoff": "300 Ry",
}

maker = RelaxMaker.fixed_cell_relaxation(
    user_params=user_params,
    dry_run=True,
)

job = maker.make(structure)
response = run_locally(job, create_folders=True)

print("\n✓ Generated FDF with global PAO.BasisSize = DZP")
print("  All Si atoms will use the same DZP basis")
print("\nCheck the FDF file for:")
print("  PAO.BasisSize    DZP")
print("\nCost vs Accuracy:")
print("  SZ  < DZ  < SZP < DZP < TZP")
print("  Fast       →→→       Accurate")
print("        (Higher cost)")
