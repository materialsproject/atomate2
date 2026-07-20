#!/usr/bin/env python
"""Basic structure format conversion using pymatgen."""

from pymatgen.core import Structure

# Read from CIF
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Write to different formats
structure.to(filename="output.xsf", fmt="xsf")
structure.to(filename="output.json", fmt="json")
structure.to(filename="POSCAR", fmt="poscar")

print("✓ Converted to XSF, JSON, and POSCAR formats")
