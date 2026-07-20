"""
Tutorial 10: Species Variants for BSSE Corrections (Ghost Atoms)

This tutorial demonstrates using species variants with ghost atoms for
Basis Set Superposition Error (BSSE) corrections via the counterpoise method.

Date: 2025-01-17
Feature: Phase 1 - Dict format for species variants
Complexity: ⭐⭐⭐⭐ Advanced

Key Concepts:
--------------
1. Ghost atoms: Atoms with Z<0 (basis functions but no nucleus/electrons)
2. Counterpoise correction: E_BSSE = E_AB - E_A(AB) - E_B(AB)
3. Same basis sets for real and ghost atoms (consistency)
4. Dict format enables clean ghost atom specification

BSSE Correction Method:
-----------------------
1. Calculate AB dimer:        E_AB
2. Calculate A in AB basis:   E_A(AB)  ← B atoms as ghosts
3. Calculate B in AB basis:   E_B(AB)  ← A atoms as ghosts
4. BSSE = E_AB - E_A(AB) - E_B(AB)
5. Corrected binding: E_bind = E_AB - E_A - E_B - BSSE

Use Case: H2O dimer
--------------------
Real atoms:  H2O molecule 1 + H2O molecule 2
Ghost atoms: H_ghost, O_ghost for counterpoise calculations

Species variants needed:
- H: Real hydrogen atoms (DZP)
- O: Real oxygen atoms (DZP)
- H_ghost: Ghost hydrogen (DZP - same basis!)
- O_ghost: Ghost oxygen (DZP - same basis!)
"""

from pymatgen.core import Structure, Lattice, Molecule
import numpy as np
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally


# Create H2O dimer structure for BSSE correction demonstration
print("=" * 70)
print("Tutorial 10: Species Variants for BSSE Corrections (Ghost Atoms)")
print("=" * 70)

print("\n📚 Background: Basis Set Superposition Error (BSSE)")
print("-" * 70)
print("When calculating binding energies, atoms in a complex use basis")
print("functions from nearby atoms, artificially lowering the energy.")
print("This 'basis set superposition error' must be corrected!")
print()
print("Counterpoise Method:")
print("  E_BSSE = E_AB - E_A(in AB basis) - E_B(in AB basis)")
print("  where ghost atoms provide basis functions without electrons")

print("\n1️⃣  Creating H2O dimer structure...")

# Create first H2O molecule
h2o1 = Molecule(
    ["O", "H", "H"],
    [
        [0.0, 0.0, 0.0],  # O
        [0.757, 0.586, 0.0],  # H
        [-0.757, 0.586, 0.0],  # H
    ],
)

# Create second H2O molecule (translated and rotated)
h2o2 = Molecule(
    ["O", "H", "H"],
    [
        [3.0, 0.0, 0.0],  # O
        [3.757, -0.586, 0.0],  # H
        [2.243, -0.586, 0.0],  # H
    ],
)

# Combine into dimer in a box
box_size = 15.0  # Å
lattice = Lattice.cubic(box_size)

# All atoms (real H2O dimer)
all_species = list(h2o1.species) + list(h2o2.species)
all_coords = list(h2o1.cart_coords) + list(h2o2.cart_coords)

# Shift to center of box
center_shift = np.array([box_size / 2, box_size / 2, box_size / 2])
all_coords_shifted = [
    coord + center_shift - np.array([1.5, 0, 0]) for coord in all_coords
]

# Create structure
structure_dimer = Structure(
    lattice,
    [str(s) for s in all_species],
    all_coords_shifted,
    coords_are_cartesian=True,
)

print("\nH2O dimer structure created:")
print(f"  Formula: {structure_dimer.composition}")
print(f"  Atoms: {len(structure_dimer)}")
print(f"  Box: {box_size} x {box_size} x {box_size} Å³")

print("\n2️⃣  Demonstrating three calculations needed for BSSE correction:")
print()

# ============================================================================
# Calculation 1: Real dimer E_AB
# ============================================================================
print("   Calculation 1: Real dimer (E_AB)")
print("   " + "-" * 66)

# No species variants needed - all real atoms
user_params_dimer = {
    "PAO.BasisSize": "DZP",  # All atoms use DZP
    "Mesh.Cutoff": "300 Ry",
    "XC.functional": "GGA",
    "XC.authors": "PBE",
}

print("      All atoms REAL:")
print("         3 O atoms  (DZP basis)")
print("         3 H atoms  (DZP basis)")
print("      → Calculates: E_AB (dimer energy)")
print()

# ============================================================================
# Calculation 2: Molecule A with ghost B: E_A(AB)
# ============================================================================
print("   Calculation 2: Molecule A + ghost B (E_A in AB basis)")
print("   " + "-" * 66)

# Create structure with first H2O real, second H2O as ghosts
species_labels_A = [
    "O",  # Molecule A: real
    "H",
    "H",
    "O_ghost",  # Molecule B: ghosts
    "H_ghost",
    "H_ghost",
]

structure_A_ghosts = structure_dimer.copy()
structure_A_ghosts.add_site_property("species_label", species_labels_A)

# IMPORTANT: Ghost atoms must have SAME basis as real atoms
user_params_A_ghosts = {
    "%block PAO.BasisSizes": {
        "O": "DZP",  # Real oxygen
        "H": "DZP",  # Real hydrogen
        "O_ghost": "DZP",  # 🔥 Ghost oxygen (SAME basis as real O!)
        "H_ghost": "DZP",  # 🔥 Ghost hydrogen (SAME basis as real H!)
    },
    "Mesh.Cutoff": "300 Ry",
    "XC.functional": "GGA",
    "XC.authors": "PBE",
}

print("      Molecule A: REAL (O, H, H)")
print("      Molecule B: GHOSTS (O_ghost, H_ghost, H_ghost)")
print()
print("      🆕 Species variants:")
for label, basis in sorted(user_params_A_ghosts["%block PAO.BasisSizes"].items()):
    ghost_flag = "👻 (ghost)" if "ghost" in label else "(real)"
    print(f"         {label:10s} → {basis:4s}  {ghost_flag}")
print()
print("      → Calculates: E_A(AB) (A in dimer basis)")
print()

# ============================================================================
# Calculation 3: Ghost A with molecule B: E_B(AB)
# ============================================================================
print("   Calculation 3: Ghost A + molecule B (E_B in AB basis)")
print("   " + "-" * 66)

# Create structure with first H2O as ghosts, second H2O real
species_labels_B = [
    "O_ghost",  # Molecule A: ghosts
    "H_ghost",
    "H_ghost",
    "O",  # Molecule B: real
    "H",
    "H",
]

structure_B_ghosts = structure_dimer.copy()
structure_B_ghosts.add_site_property("species_label", species_labels_B)

# Same basis as above (must be consistent!)
user_params_B_ghosts = user_params_A_ghosts.copy()

print("      Molecule A: GHOSTS (O_ghost, H_ghost, H_ghost)")
print("      Molecule B: REAL (O, H, H)")
print()
print("      Species variants: (same as calculation 2)")
for label, basis in sorted(user_params_B_ghosts["%block PAO.BasisSizes"].items()):
    ghost_flag = "👻 (ghost)" if "ghost" in label else "(real)"
    print(f"         {label:10s} → {basis:4s}  {ghost_flag}")
print()
print("      → Calculates: E_B(AB) (B in dimer basis)")
print()

# ============================================================================
# Summary
# ============================================================================
print("\n3️⃣  BSSE Correction Formula:")
print("   " + "-" * 66)
print("      BSSE = E_AB - E_A(AB) - E_B(AB)")
print("      E_bind(corrected) = E_AB - E_A - E_B - BSSE")
print()
print("   where:")
print("      E_AB     = Energy of real dimer (calculation 1)")
print("      E_A(AB)  = Energy of A with ghost B (calculation 2)")
print("      E_B(AB)  = Energy of B with ghost A (calculation 3)")
print("      E_A      = Energy of isolated A (separate calc)")
print("      E_B      = Energy of isolated B (separate calc)")

print("\n4️⃣  Creating example job (calculation 2: A + ghost B)...")

# Create static calculation for demonstration
maker = StaticMaker.scf(
    user_params=user_params_A_ghosts,
    # name="H2O_dimer_A_plus_ghost_B",
)

job = maker.make(structure_A_ghosts)

print("\n5️⃣  Running in dry-run mode (inspect FDF file)...")
print("   (Generating input files only - no SIESTA run)")

response = run_locally(
    job,
    create_folders=True,
    ensure_success=False,
    root_dir="dry_run_output_10_bsse_ghost_atoms",
)

print("\n" + "=" * 70)
print("✅ Tutorial Complete!")
print("=" * 70)

print("\n📁 Check the generated FDF file:")
print("   dry_run_output/10_bsse_ghost_atoms/job_*/siesta.fdf")

print("\n🔍 What to look for in siesta.fdf:")
print("   1. ChemicalSpeciesLabel block:")
print("      1   8  O        O.psml")
print("      2   1  H        H.psml")
print("      3  -8  O_ghost  O_ghost.psml  ← Negative Z!")
print("      4  -1  H_ghost  H_ghost.psml  ← Negative Z!")
print()
print("   2. Pseudopotential files (symlinked):")
print("      O_ghost.psml → O.psml (same pseudopotential)")
print("      H_ghost.psml → H.psml (same pseudopotential)")
print()
print("   3. %block PAO.BasisSizes:")
print("      O        DZP")
print("      H        DZP")
print("      O_ghost  DZP  ← Same basis as real O!")
print("      H_ghost  DZP  ← Same basis as real H!")

print("\n💡 Key Requirements for BSSE Correction:")
print("   ✅ Ghost atoms MUST have SAME basis as real atoms")
print("   ✅ Ghost atoms have negative atomic number (Z < 0)")
print("   ✅ Ghost atoms use same pseudopotentials (symlinked)")
print("   ✅ All calculations use same parameters (consistency)")

print("\n📊 Typical BSSE Magnitude:")
print("   • H-bonded complexes: 1-5 kcal/mol (~0.04-0.2 eV)")
print("   • Van der Waals:      0.5-2 kcal/mol (~0.02-0.09 eV)")
print("   • Covalent bonds:     Usually negligible")
print("   • Larger for smaller basis sets (SZ > DZ > TZ)")

print("\n🔬 Practical Workflow:")
print("   Step 1: Calculate E_AB (real dimer)")
print("   Step 2: Calculate E_A(AB) (A + ghost B) ← This tutorial")
print("   Step 3: Calculate E_B(AB) (ghost A + B)")
print("   Step 4: Calculate E_A (isolated A)")
print("   Step 5: Calculate E_B (isolated B)")
print("   Step 6: Compute BSSE and corrected binding energy")

print("\n📚 When to Use BSSE Correction:")
print("   ✅ Weak interactions (H-bonds, Van der Waals)")
print("   ✅ Smaller basis sets (SZ, DZ, SZP, DZP)")
print("   ✅ Benchmark calculations")
print("   ✅ Publication-quality results")
print("   ⚠️  May skip for large basis (TZP, QZP)")
print("   ⚠️  Less critical for strong covalent bonds")

print("\n🚀 Advanced Applications:")
print("   • Molecular complexes (dimers, trimers)")
print("   • Hydrogen bonding studies")
print("   • Van der Waals interactions")
print("   • Physisorption energies")
print("   • DNA base pairing")

print("\n💬 Note:")
print("   The species variant feature makes BSSE corrections straightforward!")
print("   Simply define ghost variants (O_ghost, H_ghost) with matching basis")
print("   sets, and ASE handles the ChemicalSpeciesLabel and pseudopotential")
print("   generation automatically.")

print("\n🎓 Further Reading:")
print("   • Boys & Bernardi, Mol. Phys. 19, 553 (1970) - Original paper")
print("   • SIESTA manual: Ghost atoms (negative Z)")
print("   • Tutorial 09: Species variants for adsorption")
