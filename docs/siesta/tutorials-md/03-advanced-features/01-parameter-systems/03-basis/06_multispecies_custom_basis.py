# \!/usr/bin/env python
"""
Tutorial: Custom Basis for Multiple Species

This tutorial shows how to define custom basis sets for different atomic
species in the same calculation (e.g., Si and O in SiO2).

Use Case:
---------
When you have a compound and want different custom basis sets for each
element type - common in oxides, alloys, or molecules.

Example:
--------
MgO with custom basis for both Mg and O.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

# Load MgO structure
structure = Structure.from_file("../../../00-structures/MgO_mp-1265_primitive.cif")

print("\n" + "=" * 70)
print("Tutorial: Multi-Species Custom Basis")
print("=" * 70)
print(f"\nStructure: {structure.composition}")
print(f"Number of atoms: {len(structure)}")

# Custom basis for both Mg and O
custom_basis_block = """Mg  3
  n=3  0  2  E  100.0  5.0  # Mg 3s: double-zeta
    6.0  4.0
  n=3  1  1  E  100.0  5.5  # Mg 3p: single-zeta
    5.5
  n=3  2  1  P  1            # Mg 3d: polarization
    5.0
O  3
  n=2  0  2  E  50.0  3.5   # O 2s: double-zeta
    4.0  2.5
  n=2  1  2  E  50.0  4.0   # O 2p: double-zeta
    4.5  3.0
  n=3  2  1  P  1            # O 3d: polarization
    4.0
"""

user_params = {
    "PAO.BasisSize": "DZP",  # Fallback (won't be used if block present)
    "%block PAO.Basis": custom_basis_block.strip().split(
        "\n"
    ),  # Must be a list of lines!
    "a2s_kpts": [6, 6, 6],
    "Mesh.Cutoff": "350 Ry",
}

maker = RelaxMaker.fixed_cell_relaxation(
    user_params=user_params,
    dry_run=True,
)

job = maker.make(structure)
response = run_locally(job, create_folders=True)

print("\n✓ Generated FDF with multi-species custom basis")
print("\nMg basis:")
print("  3s: 2 zeta (double)")
print("  3p: 1 zeta (single)")
print("  3d: polarization")
print("\nO basis:")
print("  2s: 2 zeta (double)")
print("  2p: 2 zeta (double)")
print("  3d: polarization")
print("\nCheck the FDF file for:")
print("  %block PAO.Basis")
print("    Mg  3")
print("      n=3  0  2  E ...")
print("      ...")
print("    O  3")
print("      n=2  0  2  E ...")
print("      ...")
print("  %endblock PAO.Basis")
print("\nNote: Each species defined separately in the same block")
