#!/usr/bin/env python
"""Comparison of automatic magnetic moments for different elements.

This example demonstrates the element-specific default magnetic moments
assigned by get_default_initial_magnetic_moments().

Shows:
- 3d transition metal moments (Cr, Mn, Fe, Co, Ni)
- Non-magnetic elements returning None
- How to check what moments will be assigned
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.sets.utils import get_default_initial_magnetic_moments

print("=" * 70)
print("Automatic Magnetic Moments - Element Comparison")
print("=" * 70)

# 3d transition metals with their default moments
elements_data = {
    "Cr": (24, 2.88, "BCC"),
    "Mn": (25, 8.89, "Cubic"),
    "Fe": (26, 2.87, "BCC"),
    "Co": (27, 3.54, "FCC"),
    "Ni": (28, 3.52, "FCC"),
}

print("\n3d Transition Metal Default Moments:")
print(f"{'Element':<8} {'Z':<4} {'Lattice':<8} {'Structure':<8} {'Moment (μB)':<12}")
print("-" * 52)

for elem, (z, a, crystal) in elements_data.items():
    lattice = Lattice.cubic(a)
    structure = Structure(lattice, [elem], [[0, 0, 0]])
    magmoms = get_default_initial_magnetic_moments(structure)
    moment = magmoms[0] if magmoms else 0.0
    print(f"{elem:<8} {z:<4} {a:<8.2f} {crystal:<8} {moment:<12.1f}")

# Example with Fe2O3 - multiple magnetic atoms
print("\n" + "=" * 70)
print("Complex Oxide - Fe2O3")
print("=" * 70)

lattice = Lattice.hexagonal(5.035, 13.772)
fe2o3_structure = Structure(
    lattice,
    ["Fe"] * 4 + ["O"] * 6,
    [
        [0.0, 0.0, 0.355],
        [0.0, 0.0, 0.645],
        [0.667, 0.333, 0.022],
        [0.333, 0.667, 0.978],
        [0.306, 0.0, 0.25],
        [0.0, 0.306, 0.25],
        [0.694, 0.694, 0.25],
        [0.694, 0.0, 0.75],
        [0.0, 0.694, 0.75],
        [0.306, 0.306, 0.75],
    ],
)

magmoms = get_default_initial_magnetic_moments(fe2o3_structure)
print(f"\nStructure: {fe2o3_structure.composition}")
print(f"Magnetic moments: {magmoms}")
print(f"  {sum(1 for m in magmoms if m != 0)} Fe atoms: 4.0 μB each")
print(f"  {sum(1 for m in magmoms if m == 0)} O atoms:  0.0 μB")

# Non-magnetic element example
print("\n" + "=" * 70)
print("Non-Magnetic Element - Si")
print("=" * 70)

lattice = Lattice.cubic(5.43)
si_structure = Structure(lattice, ["Si"] * 2, [[0, 0, 0], [0.25, 0.25, 0.25]])

magmoms = get_default_initial_magnetic_moments(si_structure)
print(f"\nStructure: {si_structure.composition}")
print(f"Magnetic moments: {magmoms}")
print("  Returns None - no magnetic elements detected")
print("  → Regular non-magnetic calculation will be performed")

print("\n" + "=" * 70)
print("Summary: Element-Specific Default Moments")
print("=" * 70)
print(
    """
Automatic magnetic moment assignments:

3d Transition Metals:
  Cr: 4.0 μB    Mn: 5.0 μB    Fe: 4.0 μB
  Co: 3.0 μB    Ni: 2.0 μB

4d Transition Metals:
  Mo, Tc, Ru, Rh: 1.0 μB (default)

Lanthanides & Actinides:
  Most: 1.0 μB (default)
  Gd:   7.0 μB (highest moment)

Non-magnetic elements:
  Returns None → regular non-spin-polarized calculation

Usage pattern:
  magmoms = get_default_initial_magnetic_moments(structure)
  if magmoms is not None:
      structure.add_site_property("magmom", magmoms)
      # Magnetic calculation with automatic DM.InitSpin
  else:
      # Non-magnetic calculation
"""
)
