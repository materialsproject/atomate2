"""
Tutorial 08: Species Variants for Surface Calculations

This tutorial demonstrates how to use the NEW dict format for %block PAO.BasisSizes
to assign different basis sets to surface vs bulk atoms in slab calculations.

Date: 2025-01-17
Feature: Phase 1 - Dict format for species variants
Complexity: ⭐⭐ Intermediate

Key Concepts:
--------------
1. Species variants (e.g., Ti_surface vs Ti_bulk) using dict format
2. Higher accuracy basis for surface atoms
3. Coordination with ASE's ChemicalSpeciesLabel
4. Automatic pseudopotential file generation

Use Case: TiO2 rutile (110) surface
------------------------------------
- Ti_bulk: Interior titanium atoms (DZP - standard accuracy)
- Ti_surface: Surface titanium atoms (TZP - higher accuracy)
- O_bulk: Interior oxygen atoms (DZ - standard accuracy)
- O_surface: Surface oxygen atoms (TZP - higher accuracy)

This approach is essential for:
- Surface energy calculations
- Adsorption studies
- Interface physics
- Catalysis simulations
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally
from collections import Counter


# Create TiO2 rutile (110) surface slab
print("=" * 70)
print("Tutorial 08: Species Variants for Surface Calculations")
print("=" * 70)

print("\n1️⃣  Creating TiO2 rutile (110) slab structure...")

# TiO2 rutile lattice parameters (Å)
a = 4.594
c = 2.958

# Create slab lattice (2x1x1 supercell with vacuum)
lattice = Lattice.from_parameters(
    a=2 * a,
    b=a,
    c=c + 15.0,  # 15 Å vacuum
    alpha=90,
    beta=90,
    gamma=90,
)

# Atomic positions (fractional)
# Layer 1 (bottom - bulk-like)
# Layer 2 (bulk-like)
# Layer 3 (transition to surface)
# Layer 4 (top - surface)

species = ["Ti", "Ti", "Ti", "Ti", "O", "O", "O", "O", "O", "O", "O", "O"]
coords = [
    # Ti atoms (4 layers)
    [0.0, 0.0, 0.15],  # Layer 1 - bulk
    [0.5, 0.5, 0.25],  # Layer 2 - bulk
    [0.0, 0.0, 0.35],  # Layer 3 - surface
    [0.5, 0.5, 0.45],  # Layer 4 - surface
    # O atoms (6 per 2 Ti)
    [0.305, 0.305, 0.15],  # Bulk
    [0.695, 0.695, 0.15],  # Bulk
    [0.805, 0.805, 0.25],  # Bulk
    [0.195, 0.195, 0.25],  # Bulk
    [0.305, 0.305, 0.35],  # Surface
    [0.695, 0.695, 0.35],  # Surface
    [0.805, 0.805, 0.45],  # Surface
    [0.195, 0.195, 0.45],  # Surface
]

structure = Structure(lattice, species, coords)

# Add species labels as site property
# Assign based on z-coordinate (simplified)
species_labels = []
for i, site in enumerate(structure):
    element = site.species_string
    z_frac = site.frac_coords[2]

    # Classify as surface or bulk based on z position
    if z_frac > 0.32:  # Top half
        label = f"{element}_surface"
    else:  # Bottom half
        label = f"{element}_bulk"

    species_labels.append(label)

# Add site property for species labels
structure.add_site_property("species_label", species_labels)

print(f"\nStructure composition: {structure.composition}")
print(f"Number of atoms: {len(structure)}")
print(
    f"Lattice: a={structure.lattice.a:.2f}, b={structure.lattice.b:.2f}, c={structure.lattice.c:.2f} Å"
)

# Show species labels
print("\n2️⃣  Species labels assigned:")
label_counts = Counter(species_labels)
for label, count in sorted(label_counts.items()):
    print(f"   {label:15s}: {count} atoms")

# Create relaxation maker with species-specific basis sets
print("\n3️⃣  Setting up relaxation with species-specific basis sets...")
print("\n   🆕 NEW: Dict format for %block PAO.BasisSizes")

# OLD FORMAT (would not support species variants):
# user_params = {
#     "%block PAO.BasisSizes": ["Ti DZP", "O TZP"]
#     # ❌ Can't distinguish Ti_surface from Ti_bulk
# }

# NEW FORMAT (supports species variants):
user_params = {
    "%block PAO.BasisSizes": {
        "Ti_bulk": "DZP",  # Standard accuracy for bulk Ti
        "Ti_surface": "TZP",  # 🔥 Higher accuracy for surface Ti
        "O_bulk": "DZ",  # Standard accuracy for bulk O
        "O_surface": "TZP",  # 🔥 Higher accuracy for surface O
    },
    # Other parameters
    "a2s_kpts": [4, 4, 1],  # Reduced k-points in z (slab)
    "Mesh.Cutoff": "300 Ry",
    "XC.functional": "GGA",
    "XC.authors": "PBE",
}

print("\n   Basis set assignments:")
for label, basis in user_params["%block PAO.BasisSizes"].items():
    print(f"      {label:15s} → {basis:4s}")

# Create maker
maker = RelaxMaker.fixed_cell_relaxation(
    user_params=user_params,
)

print("\n4️⃣  Creating relaxation job...")
job = maker.make(structure)

# Run in dry-run mode to inspect generated files
print("\n5️⃣  Running in dry-run mode (inspect FDF file)...")
print("   (No actual SIESTA calculation - just generate input files)")

response = run_locally(
    job,
    create_folders=True,
    ensure_success=False,
    root_dir="dry_run_output_08_surface_variants",
)

print("\n" + "=" * 70)
print("✅ Tutorial Complete!")
print("=" * 70)

print("\n📁 Check the generated FDF file:")
print("   dry_run_output/08_surface_variants/job_*/siesta.fdf")

print("\n🔍 What to look for in siesta.fdf:")
print("   1. ChemicalSpeciesLabel block (4 species):")
print("      - Ti_bulk, Ti_surface, O_bulk, O_surface")
print("   2. Pseudopotential files:")
print("      - Ti_bulk.psml, Ti_surface.psml (both symlinked from Ti.psml)")
print("      - O_bulk.psml, O_surface.psml (both symlinked from O.psml)")
print("   3. %block PAO.BasisSizes:")
print("      - Ti_bulk   DZP")
print("      - Ti_surface TZP  ← Higher accuracy!")
print("      - O_bulk    DZ")
print("      - O_surface  TZP  ← Higher accuracy!")

print("\n💡 Key Benefits:")
print("   ✅ Higher accuracy where it matters (surface)")
print("   ✅ Computational efficiency (standard basis for bulk)")
print("   ✅ Better convergence for surface properties")
print("   ✅ Essential for adsorption, catalysis, surface energy")

print("\n📚 Real-World Applications:")
print("   • Surface energy calculations")
print("   • Adsorption energy studies")
print("   • Work function calculations")
print("   • Catalysis modeling")
print("   • Interface physics")

print("\n🚀 Next Steps:")
print("   • Tutorial 09: Species variants for adsorption")
print("   • Tutorial 10: Species variants for BSSE corrections")
print("   • See 06-surfaces-and-adsorption for complete workflows")

print("\n💬 Note:")
print("   The dict format for PAO.BasisSizes is NEW in atomate2siesta v1.0.0")
print("   It enables species variants like Ti_surface, Ti_bulk without manual")
print("   intervention. ASE automatically creates ChemicalSpeciesLabel and")
print("   pseudopotential files for all variants!")
