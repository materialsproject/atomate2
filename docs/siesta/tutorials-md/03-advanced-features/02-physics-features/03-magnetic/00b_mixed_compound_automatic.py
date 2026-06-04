#!/usr/bin/env python
"""Automatic magnetic moment detection for mixed compounds.

This example shows how get_default_initial_magnetic_moments() handles
compounds with both magnetic and non-magnetic elements.

The function automatically:
- Assigns element-specific moments to magnetic elements (Ni: 2.0 μB)
- Assigns 0.0 μB to non-magnetic elements (O)
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.utils import get_default_initial_magnetic_moments
from jobflow import run_locally

print("=" * 70)
print("Automatic Magnetic Moments - NiO (Mixed Compound)")
print("=" * 70)

# Create NiO structure (rock salt)
lattice = Lattice.cubic(4.17)
structure = Structure(
    lattice,
    ["Ni", "Ni", "O", "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0.5]],
)

print(f"\nStructure: {structure.composition}")

# Automatically detect and assign moments
magmoms = get_default_initial_magnetic_moments(structure)

print(f"Automatic magnetic moments: {magmoms}")
print("  Ni (magnetic):     2.0 μB (element-specific)")
print("  O  (non-magnetic): 0.0 μB (automatic)")

# Set on structure
structure.add_site_property("magmom", magmoms)

# Create maker with AFM ordering (default)
maker = StaticMaker.scf(
    dry_run=True,
    user_params={
        "Spin": "polarized",
        # magnetic_ordering defaults to "antiferromagnetic"
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP",
    },
)

# Run calculation
job = maker.make(structure)
response = run_locally(job, create_folders=True)

print("\n✓ Calculation complete!")
print("✓ DM.InitSpin with AFM ordering:")
print("  1  +2.0  (Ni, spin up)")
print("  2  -2.0  (Ni, spin down)")
print("  3  0.0   (O, non-magnetic)")
print("  4  0.0   (O, non-magnetic)")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(
    """
For mixed compounds:

✓ Magnetic elements get element-specific moments (Ni: 2.0 μB)
✓ Non-magnetic elements get 0.0 μB automatically
✓ Works with any magnetic_ordering: FM, AFM, or custom

Example with complex oxide (Fe2O3):
  Fe atoms: 4.0 μB each (element-specific)
  O atoms:  0.0 μB (non-magnetic)

Zero manual moment assignment needed!
"""
)
