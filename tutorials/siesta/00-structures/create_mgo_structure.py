#!/usr/bin/env python3
"""Create MgO structure files for NEB tutorial.

This script creates the MgO primitive cell structure.
MgO has the rocksalt (B1) structure with space group Fm-3m.
"""

from pymatgen.core import Structure, Lattice

# MgO primitive cell (rocksalt structure)
# Space group: Fm-3m (225)
# Lattice parameter: a = 4.212 Å (experimental)

lattice = Lattice.cubic(4.212)

# Primitive rocksalt structure has 2 atoms
# Mg at (0, 0, 0) and O at (0.5, 0.5, 0.5)
species = ["Mg", "O"]
coords = [
    [0.0, 0.0, 0.0],  # Mg
    [0.5, 0.5, 0.5],  # O
]

mgo_primitive = Structure(lattice, species, coords)

# Save primitive structure
mgo_primitive.to(filename="MgO_mp-1265_primitive.cif")
print("Created MgO_mp-1265_primitive.cif")
print(f"  Composition: {mgo_primitive.composition}")
print(f"  Lattice: a = {mgo_primitive.lattice.a:.3f} Å")
print("  Space group: Fm-3m (rocksalt)")
print()

# Also create conventional cell for reference
mgo_conventional = mgo_primitive.copy()
mgo_conventional.make_supercell([2, 2, 2])
mgo_conventional.to(filename="MgO_mp-1265_conventional_standard.cif")
print("Created MgO_mp-1265_conventional_standard.cif")
print(f"  Composition: {mgo_conventional.composition}")
print(f"  Atoms: {len(mgo_conventional)} (conventional fcc cell)")
print()

print("Structures ready for NEB tutorial!")
