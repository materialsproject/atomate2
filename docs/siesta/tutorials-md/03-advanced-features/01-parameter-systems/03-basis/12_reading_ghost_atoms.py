"""
Tutorial 12: Reading Structures with Ghost Atoms

Minimal tutorial showing how to read defect structures preserving ghost atoms.

Date: 2026-01-26
Complexity: Beginner

See Also:
- Tutorial 10: BSSE ghost atoms (creating ghost structures manually)
- tutorials/02-workflows/08-defects/ for full defect workflow tutorials
"""

from pathlib import Path

print("=" * 70)
print("Tutorial 12: Reading Structures with Ghost Atoms")
print("=" * 70)

# =============================================================================
# The Problem
# =============================================================================
print("\n1. The Problem")
print("-" * 70)
print("Standard Structure.from_file() loses ghost atom information!")
print("Ghost atoms in CIF have occupancy=0.001, in FDF have negative Z.")

# =============================================================================
# The Solution
# =============================================================================
print("\n2. The Solution - Use special read functions")
print("-" * 70)

from atomate2.siesta.sets.utils.structure_io import (  # noqa: E402
    read_cif_with_ghost,
    read_siesta_with_ghost,
)

# Example files in tutorials/00-structures/
structures_dir = Path(__file__).parent.parent.parent.parent / "00-structures"
cif_file = structures_dir / "defect_structure.cif"
fdf_file = structures_dir / "defect_structure.fdf"

# Read CIF with ghost atoms
if cif_file.exists():
    structure = read_cif_with_ghost(str(cif_file))
    ghost_tags = structure.site_properties.get("ghost_tags", [])
    n_ghosts = sum(ghost_tags) if ghost_tags else 0
    print(f"\nread_cif_with_ghost('{cif_file.name}'):")
    print(f"  Atoms: {len(structure)}, Ghost atoms: {n_ghosts}")
    if n_ghosts:
        labels = structure.site_properties["species_label"]
        ghost_labels = [label for label, g in zip(labels, ghost_tags) if g]
        print(f"  Ghost labels: {ghost_labels}")

# Read FDF with ghost atoms
if fdf_file.exists():
    structure = read_siesta_with_ghost(str(fdf_file))
    ghost_tags = structure.site_properties.get("ghost_tags", [])
    n_ghosts = sum(ghost_tags) if ghost_tags else 0
    print(f"\nread_siesta_with_ghost('{fdf_file.name}'):")
    print(f"  Atoms: {len(structure)}, Ghost atoms: {n_ghosts}")

# =============================================================================
# Quick Reference
# =============================================================================
print("\n3. Quick Reference")
print("-" * 70)
print(
    """
| File Type | Function                 | Example                              |
|-----------|--------------------------|--------------------------------------|
| CIF       | read_cif_with_ghost()    | read_cif_with_ghost("defect.cif")    |
| FDF       | read_siesta_with_ghost() | read_siesta_with_ghost("defect.fdf") |
| XV        | read_siesta_with_ghost() | read_siesta_with_ghost("x.fdf", use_xv=True) |
"""
)

# =============================================================================
# Usage with Makers
# =============================================================================
print("4. Usage with Makers")
print("-" * 70)
print(
    """
from atomate2.siesta.sets.utils.structure_io import read_cif_with_ghost
from atomate2.siesta.jobs.core import RelaxMaker

structure = read_cif_with_ghost("vacancy/defect_structure.cif")
maker = RelaxMaker.fixed_cell_relaxation()
job = maker.make(structure)  # Ghost atoms handled automatically!
"""
)

print("=" * 70)
print("Done! See tutorial 10 for BSSE ghost atoms (manual creation).")
print("=" * 70)
