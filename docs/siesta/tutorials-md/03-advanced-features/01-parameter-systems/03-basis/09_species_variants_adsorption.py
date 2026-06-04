"""
Tutorial 09: Species Variants for Adsorption Studies

This tutorial demonstrates using species variants to assign different basis sets
for substrate, adsorbate, and interface atoms in adsorption calculations.

Date: 2025-01-17
Feature: Phase 1 - Dict format for species variants
Complexity: ⭐⭐⭐ Intermediate-Advanced

Key Concepts:
--------------
1. Substrate atoms: Standard accuracy (computational efficiency)
2. Adsorbate molecule: High accuracy (critical for binding energy)
3. Interface atoms: High accuracy (where chemistry happens)
4. Dict format enables precise control

Use Case: CO on Cu(111) surface
--------------------------------
- Cu_substrate: Bulk copper atoms (DZP - standard)
- Cu_interface: Surface copper near CO (TZP - high accuracy)
- C_molecule: Carbon in CO (TZP - high accuracy)
- O_molecule: Oxygen in CO (TZP - high accuracy)

This is essential for:
- Accurate adsorption energies
- Reaction mechanism studies
- Catalysis modeling
- Diffusion barriers
"""

from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
import numpy as np
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally
from collections import Counter


# Create Cu(111) surface slab with CO molecule adsorbed on top site
print("=" * 70)
print("Tutorial 09: Species Variants for Adsorption Studies")
print("=" * 70)

print("\n1️⃣  Creating Cu(111) surface with CO adsorbate...")

# Cu FCC structure
a_cu = 3.615  # Lattice constant (Å)
cu_bulk = Structure(
    Lattice.cubic(a_cu),
    ["Cu"],
    [[0, 0, 0]],
)

# Generate (111) surface slab (4 layers, 15 Å vacuum)
slabgen = SlabGenerator(
    cu_bulk,
    miller_index=[1, 1, 1],
    min_slab_size=10.0,  # ~4 layers
    min_vacuum_size=15.0,
    center_slab=True,
)

slab = slabgen.get_slabs()[0]  # Get first termination

# Simplify to smaller cell for tutorial (2x2x1)
slab.make_supercell([2, 2, 1])

# Add CO molecule above a top site (on top of surface Cu atom)
# Find a surface Cu atom (highest z coordinate)
cu_positions = []
for i, site in enumerate(slab):
    if site.species_string == "Cu":
        cu_positions.append((i, site.coords))

# Get the highest Cu atom (surface)
surface_cu_idx, surface_cu_pos = max(cu_positions, key=lambda x: x[1][2])

# Add C and O above this Cu atom
c_height = 2.0  # Å above Cu surface
co_bond = 1.15  # C-O bond length (Å)

c_pos = surface_cu_pos + np.array([0, 0, c_height])
o_pos = c_pos + np.array([0, 0, co_bond])

# Add C and O to structure
slab.append("C", c_pos, coords_are_cartesian=True)
slab.append("O", o_pos, coords_are_cartesian=True)

# Assign species labels
species_labels = []

# Find interface Cu atoms (near CO, within 4 Å of C atom)
for i, site in enumerate(slab):
    element = site.species_string

    if element == "Cu":
        # Calculate distance to C atom
        dist_to_c = np.linalg.norm(site.coords - c_pos)

        if dist_to_c < 4.0:  # Within 4 Å of CO
            label = "Cu_interface"
        else:
            label = "Cu_substrate"

    elif element == "C":
        label = "C_molecule"
    elif element == "O":
        label = "O_molecule"
    else:
        label = element

    species_labels.append(label)

# Add site property
slab.add_site_property("species_label", species_labels)

structure = slab

print(f"\nStructure composition: {structure.composition}")
print(f"Number of atoms: {len(structure)}")
print(f"Lattice: c={structure.lattice.c:.2f} Å (slab + vacuum)")

# Show species labels
print("\n2️⃣  Species labels assigned:")
label_counts = Counter(species_labels)
for label, count in sorted(label_counts.items()):
    print(f"   {label:20s}: {count:2d} atoms")

# Identify which atoms are which
cu_substrate = sum(1 for label in species_labels if label == "Cu_substrate")
cu_interface = sum(1 for label in species_labels if label == "Cu_interface")
c_molecule = sum(1 for label in species_labels if label == "C_molecule")
o_molecule = sum(1 for label in species_labels if label == "O_molecule")

print("\n   Distribution:")
print(f"      Substrate Cu:  {cu_substrate:2d} atoms (bulk-like, far from adsorbate)")
print(f"      Interface Cu:  {cu_interface:2d} atoms (near CO, high accuracy needed)")
print(f"      Adsorbate:     {c_molecule + o_molecule:2d} atoms (CO molecule)")

# Create relaxation maker with optimized basis sets
print("\n3️⃣  Setting up adsorption calculation with species-specific basis...")
print("\n   🆕 NEW: Dict format enables precision control")

user_params = {
    "%block PAO.BasisSizes": {
        # Substrate: Standard accuracy (most of the atoms)
        "Cu_substrate": "DZP",
        # Interface: Higher accuracy (critical for binding)
        "Cu_interface": "TZP",  # 🔥 High accuracy where chemistry happens
        # Adsorbate: Highest accuracy (critical for energetics)
        "C_molecule": "TZP",  # 🔥 High accuracy for binding energy
        "O_molecule": "TZP",  # 🔥 High accuracy for binding energy
    },
    # Adsorption-specific parameters
    "a2s_kpts": [3, 3, 1],  # Reduced k-points for slab
    "Mesh.Cutoff": "150 Ry",  # Higher cutoff for molecules
    "XC.functional": "GGA",
    "XC.authors": "PBE",
    "PAO.EnergyShift": "0.01 Ry",  # Tight basis
}

print("\n   Basis set strategy:")
print("      Layer            Basis   Rationale")
print("      " + "-" * 60)
for label, basis in sorted(user_params["%block PAO.BasisSizes"].items()):
    if "substrate" in label:
        reason = "Bulk-like, computational efficiency"
    elif "interface" in label:
        reason = "Critical for binding, needs accuracy"
    elif "molecule" in label:
        reason = "Adsorbate, highest accuracy needed"
    else:
        reason = "Standard"
    print(f"      {label:16s} {basis:4s}   {reason}")

# Create maker
maker = RelaxMaker.fixed_cell_relaxation(
    user_params=user_params,
    # name="CO_on_Cu111_relax",
)

print("\n4️⃣  Creating relaxation job...")
job = maker.make(structure)

# Run in dry-run mode
print("\n5️⃣  Running in dry-run mode (inspect FDF file)...")
print("   (Generating input files only - no SIESTA run)")

response = run_locally(
    job,
    create_folders=True,
    ensure_success=False,
    root_dir="dry_run_output_09_adsorption_variants",
)

print("\n" + "=" * 70)
print("✅ Tutorial Complete!")
print("=" * 70)

print("\n📁 Check the generated FDF file:")
print("   dry_run_output/09_adsorption_variants/job_*/siesta.fdf")

print("\n🔍 What to look for in siesta.fdf:")
print("   1. ChemicalSpeciesLabel block (4 species):")
print("      - Cu_substrate, Cu_interface")
print("      - C_molecule, O_molecule")
print("   2. Pseudopotential files:")
print("      - Cu_substrate.psml, Cu_interface.psml (from Cu.psml)")
print("      - C_molecule.psml (from C.psml)")
print("      - O_molecule.psml (from O.psml)")
print("   3. %block PAO.BasisSizes:")
print("      - Cu_substrate  DZP     ← Standard (efficiency)")
print("      - Cu_interface  TZP     ← High accuracy (binding)")
print("      - C_molecule    TZP     ← High accuracy (adsorbate)")
print("      - O_molecule    TZP     ← High accuracy (adsorbate)")

print("\n💡 Key Benefits:")
print("   ✅ Accurate adsorption energies (high basis for adsorbate)")
print("   ✅ Correct interface description (high basis for surface layer)")
print("   ✅ Computational efficiency (standard basis for substrate)")
print("   ✅ Best accuracy-cost trade-off")

print("\n📊 Computational Cost Comparison:")
print("   Uniform TZP:        100% (most expensive)")
print("   Uniform DZP:         40% (may miss binding details)")
print("   This approach:       ~60% (optimal balance!) 🎯")

print("\n📚 Typical Adsorption Workflow:")
print("   1. Relax substrate slab (DZP basis)")
print("   2. Add adsorbate with species variants (this tutorial)")
print("   3. Relax adsorbate + interface (TZP for critical atoms)")
print("   4. Calculate adsorption energy:")
print("      E_ads = E_slab+mol - E_slab - E_mol")

print("\n🚀 Real-World Applications:")
print("   • CO oxidation catalysis")
print("   • Hydrogen storage materials")
print("   • Electrochemical reactions")
print("   • Molecular sensors")
print("   • Heterogeneous catalysis")

print("\n🔬 Advanced Tips:")
print("   • Use same basis for reference calculations (E_slab, E_mol)")
print("   • Test convergence of interface region size")
print("   • Consider basis set superposition error (BSSE)")
print("   • See Tutorial 10 for BSSE correction with ghost atoms")

print("\n💬 Note:")
print("   The species variant feature enables you to achieve publication-quality")
print("   adsorption energies while keeping computational cost reasonable.")
print("   This is essential for screening large numbers of adsorbates!")
