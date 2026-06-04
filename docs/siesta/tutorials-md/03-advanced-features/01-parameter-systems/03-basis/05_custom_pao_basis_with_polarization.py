# \!/usr/bin/env python
"""
Tutorial: Custom PAO.Basis with Polarization Orbitals

This tutorial shows how to add polarization orbitals to your custom basis.
Polarization orbitals (d for sp elements, p for s elements) improve accuracy
for bonding and charge transfer.

Use Case:
---------
When you want custom control over basis size AND include polarization
for better description of chemical bonding.

Example:
--------
Silicon with custom s,p orbitals + d polarization.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

# Load silicon structure
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

print("\n" + "=" * 70)
print("Tutorial: Custom PAO.Basis with Polarization")
print("=" * 70)
print(f"\nStructure: {structure.composition}")
print(f"Number of atoms: {len(structure)}")

# Custom basis with polarization (3d orbital for Si)
custom_basis_block = """Si  3
  n=3  0  2  E  50.0  4.5  # 3s: double-zeta
    5.0  3.5
  n=3  1  2  E  50.0  5.0  # 3p: double-zeta
    5.5  4.0
  n=3  2  1  P  1  # 3d: polarization orbital
    5.0
"""

user_params = {
    "PAO.BasisSize": "DZP",  # This will be overridden by the block
    "%block PAO.Basis": custom_basis_block.strip().split(
        "\n"
    ),  # Must be a list of lines!
    "a2s_kpts": [6, 6, 6],
    "Mesh.Cutoff": "300 Ry",
}

maker = RelaxMaker.fixed_cell_relaxation(
    user_params=user_params,
    dry_run=True,
)

job = maker.make(structure)
response = run_locally(job, create_folders=True)

print("\n✓ Generated FDF with custom polarized basis")
print("\nCustom basis details:")
print("  Si 3s: 2 zeta (double-zeta)")
print("  Si 3p: 2 zeta (double-zeta)")
print("  Si 3d: 1 zeta (polarization)")
print("\nCheck the FDF file for:")
print("  %block PAO.Basis")
print("    Si  3                # 3 shells (s, p, d)")
print("      n=3  0  2  E ...   # s orbital")
print("      n=3  1  2  E ...   # p orbital")
print("      n=3  2  1  P  1    # d polarization")
print("  %endblock PAO.Basis")
print("\nKey points:")
print("  - P flag indicates polarization orbital")
print("  - d orbitals (l=2) are common polarization for Si")
print("  - This gives DZP-like quality with custom control")
