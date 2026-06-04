#!/usr/bin/env python
"""
Tutorial: Custom PAO.Basis Block - Simple Example

This tutorial shows how to use %block PAO.Basis to fully customize the
basis set for specific atoms. This gives you complete control over the
number of zeta functions and their cutoff radii.

Use Case:
---------
When you need very specific control over the basis functions, beyond
what standard sizes (SZ, DZ, TZP) provide.

Example:
--------
Customizing Silicon atom 1 with specific orbital configuration.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

# Load silicon structure
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

print("\n" + "=" * 70)
print("Tutorial: Custom PAO.Basis Block (Simple)")
print("=" * 70)
print(f"\nStructure: {structure.composition}")
print(f"Number of atoms: {len(structure)}")

# Define custom basis for Silicon
# Format: species, l-shell, number-of-zetas, [cutoff radii...]
custom_basis_block = """Si  2
  n=3  0  2  E  50.0  4.5  # 3s orbital: 2 zeta, cutoffs at 4.5 and auto
    5.0  3.5
  n=3  1  1  E  50.0  5.0  # 3p orbital: 1 zeta, cutoff at 5.0
    5.5
"""

user_params = {
    "PAO.BasisSize": "DZP",  # Global default for other atoms
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

print("\n✓ Generated FDF with custom PAO.Basis block")
print("\nCustom basis details:")
print("  Si 3s orbital: 2 zeta functions (cutoffs: 5.0, 3.5 Bohr)")
print("  Si 3p orbital: 1 zeta function (cutoff: 5.5 Bohr)")
print("\nCheck the FDF file for:")
print("  %block PAO.Basis")
print("    Si  2")
print("      n=3  0  2  E  50.0  4.5")
print("        5.0  3.5")
print("      n=3  1  1  E  50.0  5.0")
print("        5.5")
print("  %endblock PAO.Basis")
print("\nExplanation of format:")
print("  n=3  0  2    → n=3 (shell), l=0 (s orbital), 2 zeta functions")
print("  E  50.0  4.5 → Energy shift 50 meV, split norm 4.5")
print("  5.0  3.5     → Cutoff radii for the 2 zeta functions (Bohr)")
