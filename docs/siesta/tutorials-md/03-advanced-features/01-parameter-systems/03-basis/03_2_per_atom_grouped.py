#!/usr/bin/env python
"""
Tutorial 03.2: Per-Atom Basis - Grouped Specification

This tutorial demonstrates per-atom basis control using logical groups.
Perfect for layer-based systems (surfaces, interfaces, multilayers).

Use Case:
---------
Surface slabs with well-defined layers:
- Surface layer: atoms 1, 4, 5 → TZP (highest accuracy)
- Subsurface layer: atoms 2, 6, 7 → DZP (medium accuracy)
- Bulk layer: atoms 3, 8, 9 → DZ (efficient)

Key Feature (NEW in v1.0.0):
----------------------------
Grouped specification provides cleaner code for layer-based systems.
Helper function converts groups → per-atom dict → species labels + PAO.BasisSizes.

Author: Arsalan Akhtar
Date: 2025-12-14
"""

from pathlib import Path

from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.utils import create_per_atom_basis_dict

# Create output directory
output_dir = Path("dry_run_output_03_2_per_atom_grouped")
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("Tutorial 03.2: Per-Atom Basis - Grouped Specification")
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
# Grouped Atom Specification (Layer-Based)
# ============================================================================
print("\n" + "=" * 70)
print("Grouped Atom Specification (Layer-Based)")
print("=" * 70)

print("\n📋 Defining atoms by logical groups:")

atom_groups = {
    "surface": ([1, 4, 5], "TZP"),  # Surface atoms: highest accuracy
    "subsurface": ([2, 6, 7], "DZP"),  # Subsurface: medium accuracy
    "bulk": ([3, 8, 9], "DZ"),  # Bulk: efficient basis
}

print("\n   atom_groups = {")
for group_name, (atom_indices, basis) in atom_groups.items():
    print(f"       '{group_name}': ({atom_indices}, '{basis}'),")
print("   }")

print("\n   Rationale:")
print("      • Surface layer: Needs high accuracy (exposed, reactive)")
print("      • Subsurface: Medium accuracy (interface effects)")
print("      • Bulk: Efficient basis (bulk-like, less critical)")

print("\n2️⃣  Creating per-atom basis from groups...")

# Create per-atom basis from groups
species_labels, pao_basissizes = create_per_atom_basis_dict(structure, atom_groups)

print(f"\n✅ Generated {len(set(species_labels))} unique species:")
for label, basis in sorted(pao_basissizes.items()):
    count = species_labels.count(label)
    atoms = [i + 1 for i, lbl in enumerate(species_labels) if lbl == label]
    print(f"      {label:12s} : {basis:5s} ({count} atoms: {atoms})")

print("\n   Species label assignment:")
for i, label in enumerate(species_labels, start=1):
    element = structure[i - 1].species_string
    # Find which group this atom belongs to
    group_name = "unknown"
    for gname, (indices, _) in atom_groups.items():
        if i in indices:
            group_name = gname
            break
    print(f"      Atom {i:2d} ({element:2s}): {label:12s}  [{group_name}]")

# Add to structure as site property
structure.add_site_property("species_label", species_labels)

print("\n3️⃣  Creating RelaxMaker with grouped specification...")

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
# Advanced Example: Automatic Layer Detection
# ============================================================================
print("\n" + "=" * 70)
print("Advanced: Automatic Layer Detection by Z-Coordinate")
print("=" * 70)

print("\n💡 Pro Tip: Automatically assign atoms to groups based on position")

# Get z-coordinates of all atoms
z_coords = [site.frac_coords[2] for site in structure]

# Define thresholds
surface_threshold = 0.10
subsurface_threshold = 0.20

# Automatically assign to groups
auto_groups = {"surface": ([], "TZP"), "subsurface": ([], "DZP"), "bulk": ([], "DZ")}

for i, z in enumerate(z_coords, start=1):
    if z < surface_threshold:
        auto_groups["surface"][0].append(i)
    elif z < subsurface_threshold:
        auto_groups["subsurface"][0].append(i)
    else:
        auto_groups["bulk"][0].append(i)

# Convert to proper format
auto_groups_formatted = {
    name: (indices, basis) for name, (indices, basis) in auto_groups.items()
}

print("\n   Automatic grouping by z-coordinate:")
for group_name, (atom_indices, basis) in auto_groups_formatted.items():
    if atom_indices:
        print(f"      {group_name:12s}: atoms {atom_indices} → {basis}")

print("\n   Code pattern:")
print(
    """
   z_coords = [site.frac_coords[2] for site in structure]

   auto_groups = {"surface": ([], "TZP"), "bulk": ([], "DZ")}
   for i, z in enumerate(z_coords, start=1):
       if z < threshold:
           auto_groups["surface"][0].append(i)
       else:
           auto_groups["bulk"][0].append(i)

   labels, basis_dict = create_per_atom_basis_dict(structure, auto_groups)
"""
)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary: Grouped Specification")
print("=" * 70)

print("\n✅ Advantages:")
print("   • Cleaner code - logical grouping (surface/bulk/interface)")
print("   • Perfect for slabs, interfaces, multilayers")
print("   • Less prone to indexing errors")
print("   • Can automate group assignment (e.g., by z-coordinate)")
print("   • Validation: checks for overlapping groups")

print("\n🎯 Use Cases:")
print("   • Surface slabs: surface/subsurface/bulk layers")
print("   • Interfaces: material_A/interface/material_B")
print("   • Heterostructures: layer1/layer2/layer3")
print("   • Multilayers: alternating accuracy by layer")
print("   • Adsorption: adsorbate/interface/substrate")

print("\n💡 Code Pattern:")
print(
    """
   from atomate2.siesta.sets.utils import create_per_atom_basis_dict

   # Define logical groups
   atom_groups = {
       "surface": ([1, 2, 3], "TZP"),    # High accuracy
       "bulk": ([4, 5, 6], "DZ"),        # Efficient
   }

   # Apply and get species labels + basis dict
   labels, basis_dict = create_per_atom_basis_dict(structure, atom_groups)

   # Add to structure
   structure.add_site_property("species_label", labels)

   # Use in RelaxMaker
   maker = RelaxMaker(user_params={'%block PAO.BasisSizes': basis_dict})
"""
)

print("\n⚠️  Important:")
print("   • Atom indices are 1-indexed (like SIESTA)")
print("   • Groups must not overlap (raises ValueError)")
print("   • Ungrouped atoms use fallback_basis (default: DZP)")

print("\n" + "=" * 70)
print("Tutorial Complete! ✨")
print("=" * 70)
print(f"\n📁 Check FDF file: {output_dir}/job_*/siesta.fdf")
print("   Look for:")
print("      • %block ChemicalSpeciesLabel (species variants)")
print("      • %block PAO.BasisSizes (6 unique species)")
print("\n📚 Related:")
print("   • Tutorial 03.1: Direct per-atom specification")
print("   • Tutorial 08: Species variants for surfaces")
print("   • Tutorial 11: PAO.Basis helper functions")
