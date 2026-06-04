#!/usr/bin/env python
"""
Tutorial 03.1: Per-Atom Basis - Direct Specification

This tutorial demonstrates per-atom basis control using direct atom indices.
Perfect for systems requiring precise atom-by-atom control.

Use Case:
---------
Surface slabs where you need different basis sizes for specific atoms:
- Surface layer atoms (high accuracy - TZP)
- Subsurface atoms (medium accuracy - DZP)
- Bulk atoms (efficient - DZ)

Key Feature (NEW in v1.0.0):
----------------------------
Helper function automatically creates species labels and PAO.BasisSizes dict
from per-atom specifications (1-indexed like SIESTA).

Author: Arsalan Akhtar
Date: 2025-12-14
"""

from pathlib import Path

from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.utils import apply_per_atom_basis

# Create output directory
output_dir = Path("dry_run_output_03_1_per_atom_direct")
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("Tutorial 03.1: Per-Atom Basis - Direct Specification")
print("=" * 70)

# Create TiO2 surface slab (simplified 9-atom structure)
# Layers: surface (top) → subsurface → bulk (bottom)
lattice = Lattice.from_parameters(10.0, 10.0, 20.0, 90, 90, 90)
structure = Structure(
    lattice,
    ["Ti", "Ti", "Ti", "O", "O", "O", "O", "O", "O"],
    [
        [0.5, 0.5, 0.05],  # Ti atom 1 - surface
        [0.0, 0.0, 0.15],  # Ti atom 2 - subsurface
        [0.5, 0.5, 0.25],  # Ti atom 3 - bulk
        [0.25, 0.25, 0.08],  # O atom 4 - surface
        [0.75, 0.75, 0.08],  # O atom 5 - surface
        [0.25, 0.75, 0.18],  # O atom 6 - subsurface
        [0.75, 0.25, 0.18],  # O atom 7 - subsurface
        [0.25, 0.25, 0.28],  # O atom 8 - bulk
        [0.75, 0.75, 0.28],  # O atom 9 - bulk
    ],
)

print(f"\n1️⃣  Structure created: {structure.composition}")
print(f"   Total atoms: {len(structure)}")
print("\n   Layer distribution:")
print("      Surface:    Ti(1), O(4,5)     - z > 0.07")
print("      Subsurface: Ti(2), O(6,7)     - 0.14 < z < 0.20")
print("      Bulk:       Ti(3), O(8,9)     - z > 0.24")

# ============================================================================
# Direct Per-Atom Specification (Expert Control)
# ============================================================================
print("\n" + "=" * 70)
print("Direct Per-Atom Basis Specification")
print("=" * 70)

print("\n📋 Defining basis for each atom individually (1-indexed):")
per_atom_basis = {
    1: "TZP",  # Ti atom 1 (surface) - highest accuracy
    2: "TZP",  # Ti atom 2 (subsurface) - high accuracy
    3: "DZP",  # Ti atom 3 (bulk) - medium accuracy
    4: "TZP",  # O atom 4 (surface)
    5: "TZP",  # O atom 5 (surface)
    6: "DZP",  # O atom 6 (subsurface)
    7: "DZP",  # O atom 7 (subsurface)
    # Atoms 8-9 use fallback (DZ)
}

print("\n   per_atom_basis = {")
for atom_idx, basis in per_atom_basis.items():
    element = structure[atom_idx - 1].species_string
    z_frac = structure[atom_idx - 1].frac_coords[2]
    print(f"       {atom_idx}: '{basis}',  # {element} atom at z={z_frac:.2f}")
print("       # Atoms 8-9 use fallback (DZ)")
print("   }")

print("\n2️⃣  Applying per-atom basis using helper function...")

# Apply per-atom basis using helper function
species_labels, pao_basissizes = apply_per_atom_basis(
    structure, per_atom_basis, fallback_basis="DZ"
)

print(f"\n✅ Generated {len(set(species_labels))} unique species:")
for label, basis in sorted(pao_basissizes.items()):
    count = species_labels.count(label)
    print(f"      {label:12s} : {basis:5s} ({count} atoms)")

print("\n   Species label assignment:")
for i, label in enumerate(species_labels, start=1):
    element = structure[i - 1].species_string
    print(f"      Atom {i:2d} ({element:2s}): {label}")

# Add to structure as site property
structure.add_site_property("species_label", species_labels)

print("\n3️⃣  Creating RelaxMaker with per-atom basis...")

user_params = {
    "%block PAO.BasisSizes": pao_basissizes,
    "Mesh.Cutoff": "300 Ry",
    "a2s_kpts": [4, 4, 1],
}

print("\n   user_params = {")
print(f"       '%block PAO.BasisSizes': {pao_basissizes},")
print(f"       'Mesh.Cutoff': '{user_params['Mesh.Cutoff']}',")
print(f"       'a2s_kpts': {user_params['a2s_kpts']},")
print("   }")

maker = RelaxMaker.fixed_cell_relaxation(user_params=user_params, dry_run=True)

job = maker.make(structure)
response = run_locally(job, create_folders=True, root_dir=str(output_dir))

print("\n4️⃣  Calculation completed (dry-run mode)")
print(f"   📁 Output: {output_dir}/job_*/siesta.fdf")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary: Direct Per-Atom Specification")
print("=" * 70)

print("\n✅ Advantages:")
print("   • Maximum control - specify each atom individually")
print("   • Perfect for irregular structures, defects, dopants")
print("   • Fallback basis for unspecified atoms")
print("   • Validation: checks for valid indices (1 to n_atoms)")

print("\n⚠️  Considerations:")
print("   • More verbose - must list every special atom")
print("   • Need to track atom indices carefully")
print("   • For layer-based systems, see Tutorial 03.2 (grouped method)")

print("\n🎯 Use Cases:")
print("   • Defects: High accuracy around defect site")
print("   • Dopants: Special treatment for specific doped atoms")
print("   • Asymmetric systems: Different atoms need different basis")
print("   • Fine-tuned control: Optimize basis for each atom individually")

print("\n💡 Code Pattern:")
print(
    """
   from atomate2.siesta.sets.utils import apply_per_atom_basis

   # Define basis for specific atoms (1-indexed)
   per_atom_basis = {
       1: 'TZP',  # Atom 1
       2: 'DZP',  # Atom 2
       # Rest use fallback
   }

   # Apply and get species labels + basis dict
   labels, basis_dict = apply_per_atom_basis(
       structure, per_atom_basis, fallback_basis='DZ'
   )

   # Add to structure
   structure.add_site_property("species_label", labels)

   # Use in RelaxMaker
   maker = RelaxMaker(user_params={'%block PAO.BasisSizes': basis_dict})
"""
)

print("\n" + "=" * 70)
print("Tutorial Complete! ✨")
print("=" * 70)
print(f"\n📁 Check FDF file: {output_dir}/job_*/siesta.fdf")
print("   Look for:")
print("      • %block ChemicalSpeciesLabel (species variants)")
print("      • %block PAO.BasisSizes (per-species basis)")
print("\n📚 Next: Tutorial 03.2 (grouped specification for layers)")
