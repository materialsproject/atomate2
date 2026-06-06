# \!/usr/bin/env python
"""
Tutorial: Combining PAO.BasisSizes and PAO.Basis

This tutorial shows how to use BOTH %block PAO.BasisSizes and %block PAO.Basis
together for maximum flexibility.

Use Case:
---------
When you want:
- Custom basis for some species (PAO.Basis)
- Standard sizes for others (PAO.BasisSize/PAO.BasisSizes)

Example:
--------
MgO where:
- Oxygen uses custom basis (full control)
- Magnesium uses standard TZP (convenient)
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

# Load MgO structure
structure = Structure.from_file("../../../00-structures/MgO_mp-1265_primitive.cif")

print("\n" + "=" * 70)
print("Tutorial: Combining PAO.BasisSizes and PAO.Basis")
print("=" * 70)
print(f"\nStructure: {structure.composition}")
print(f"Number of atoms: {len(structure)}")

# Custom basis ONLY for Oxygen
# Mg will use standard basis from PAO.BasisSize
custom_basis_block = """O  3
  n=2  0  2  E  50.0  3.5   # O 2s: double-zeta
    4.0  2.5
  n=2  1  2  E  50.0  4.0   # O 2p: double-zeta
    4.5  3.0
  n=3  2  1  P  1            # O 3d: polarization
    4.0
"""

user_params = {
    "PAO.BasisSize": "TZP",  # Global: Mg will use this
    "%block PAO.Basis": custom_basis_block.strip().split(
        "\n"
    ),  # O uses this (must be list!)
    "a2s_kpts": [6, 6, 6],
    "Mesh.Cutoff": "350 Ry",
}

maker = RelaxMaker.fixed_cell_relaxation(
    user_params=user_params,
    dry_run=True,
)

job = maker.make(structure)
response = run_locally(job, create_folders=True)

print("\n✓ Generated FDF with mixed basis specification")
print("\nBasis assignment:")
print("  O:  Custom basis (from PAO.Basis block)")
print("      - Full control over orbitals and radii")
print("  Mg: Standard TZP (from PAO.BasisSize)")
print("      - Convenient, well-tested")
print("\nCheck the FDF file for:")
print("  PAO.BasisSize    TZP")
print("  %block PAO.Basis")
print("    O  3")
print("      n=2  0  2  E ...")
print("      ...")
print("  %endblock PAO.Basis")
print("\nBest of both worlds:")
print("  - Custom control where needed (O)")
print("  - Convenience where sufficient (Mg)")
print("  - SIESTA handles the rest automatically")
