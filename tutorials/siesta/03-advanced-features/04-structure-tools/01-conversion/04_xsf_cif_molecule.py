#!/usr/bin/env python
"""Read XSF/CIF files and molecules using siesta_to_pymatgen."""

from pymatgen.core import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from atomate2.siesta.powerups import siesta_to_pymatgen
from ase.io import write

# Create test files
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")
structure.to(filename="test.xsf", fmt="xsf")
structure.to(filename="test.cif", fmt="cif")

# Read XSF file as Structure
struct_from_xsf = siesta_to_pymatgen("test.xsf")
print(f"✓ Read XSF as Structure: {struct_from_xsf.composition}")

# Read CIF file as Structure
struct_from_cif = siesta_to_pymatgen("test.cif")
print(f"✓ Read CIF as Structure: {struct_from_cif.composition}")

# Create a molecule and write to XSF using ASE
molecule = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.2]])
atoms = AseAtomsAdaptor.get_atoms(molecule)
write("CO.xsf", atoms)

# Read XSF as Molecule
mol_from_xsf = siesta_to_pymatgen("CO.xsf", as_molecule=True)
print(f"✓ Read XSF as Molecule: {mol_from_xsf.composition}")
print(f"  Sites: {len(mol_from_xsf.sites)}")
