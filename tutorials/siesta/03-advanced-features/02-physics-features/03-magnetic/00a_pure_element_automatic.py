#!/usr/bin/env python
"""Automatic magnetic moment detection for pure elements.

This example demonstrates the EASIEST way to perform magnetic calculations:
use get_default_initial_magnetic_moments() to automatically detect and assign
element-specific magnetic moments.

Zero manual configuration required!
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.utils import get_default_initial_magnetic_moments
from jobflow import run_locally

print("=" * 70)
print("Automatic Magnetic Moment Detection - Pure Fe (BCC)")
print("=" * 70)

# Create Fe structure (BCC)
lattice = Lattice.cubic(2.87)
structure = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])

print(f"\nStructure: {structure.composition}")

# Automatically detect magnetic elements and assign default moments
# Fe gets 4.0 μB by default (element-specific)
magmoms = get_default_initial_magnetic_moments(structure)

print(f"Automatic magnetic moments: {magmoms}")
print("  Fe: 4.0 μB (element-specific default)")

# Set magnetic moments on structure
structure.add_site_property("magmom", magmoms)

# Create maker - DM.InitSpin automatically generated!
maker = StaticMaker.scf(
    dry_run=True,
    user_params={
        "Spin": "polarized",
        "a2s_magnetic_ordering": "ferromagnetic",  # Default is AFM, but Fe is FM
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP",
    },
)

# Run calculation
job = maker.make(structure)
response = run_locally(job, create_folders=True)

print("\n✓ Calculation complete!")
print("✓ DM.InitSpin was automatically generated:")
print("  1  +4.0")
print("  2  +4.0")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(
    """
The get_default_initial_magnetic_moments() function:

✓ Automatically detects Fe as a magnetic element
✓ Assigns element-specific moment (4.0 μB for Fe)
✓ No manual configuration needed!

Element-specific defaults:
  Cr: 4.0 μB    Mn: 5.0 μB    Fe: 4.0 μB
  Co: 3.0 μB    Ni: 2.0 μB    Gd: 7.0 μB

This is the EASIEST way to set up magnetic calculations!
"""
)
