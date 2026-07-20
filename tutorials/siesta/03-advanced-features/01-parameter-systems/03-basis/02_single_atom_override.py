#!/usr/bin/env python
"""
Tutorial: Per-Species Basis Sizes (%block PAO.BasisSizes)

This tutorial shows how to assign different basis sizes to different
atomic SPECIES (elements) in a compound, rather than using a global setting.

Use Case:
---------
In compounds (e.g., SiO2, MgO), you often want different basis sizes
for different elements - heavier for active elements, lighter for others.

Note: This is per-SPECIES, not per-atom. All Si atoms get one size,
all O atoms get another size.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

# Load MgO structure (has 2 species: Mg and O)
structure = Structure.from_file("../../../00-structures/MgO_mp-1265_primitive.cif")

print("\n" + "=" * 70)
print("Tutorial: Per-Species Basis Sizes")
print("=" * 70)
print(f"\nStructure: {structure.composition}")
print(f"Number of atoms: {len(structure)}")
print("Species: Mg, O")

# Different basis sizes for Mg vs O
user_params = {
    "%block PAO.BasisSizes": ["Mg DZP", "O TZP"],
    "a2s_kpts": [6, 6, 6],
    "Mesh.Cutoff": "350 Ry",
}

maker = RelaxMaker.fixed_cell_relaxation(
    user_params=user_params,
    dry_run=True,
)

job = maker.make(structure)
response = run_locally(job, create_folders=True)

print("\n✓ Generated FDF with per-species basis sizes")
print("  All Mg atoms: DZP")
print("  All O atoms:  TZP (higher accuracy for oxygen)")
print("\nCheck the FDF file for:")
print("  %block PAO.BasisSizes")
print("    Mg  DZP")
print("    O   TZP")
print("  %endblock PAO.BasisSizes")
print("\n⚠️  Note: This is per-SPECIES (element type), not per-atom!")
print("    All atoms of same species get same basis size.")
