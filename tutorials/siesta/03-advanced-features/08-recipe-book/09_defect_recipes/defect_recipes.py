#!/usr/bin/env python
"""Defect Recipes - Point defect calculations with one-liners.

NOTE: This tutorial demonstrates the defect recipe API. The underlying defect
workflows require chemical potentials for formation energy calculations, which
are not yet fully integrated into the recipe API. This tutorial is included for
completeness but may require manual parameter passing for full functionality.
"""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook

# Create MgO structure
lattice = Lattice.cubic(4.212)
mgo = Structure(
    lattice,
    ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"],
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ],
)

print("=" * 80)
print("DEFECT RECIPES - One-Liner API")
print("=" * 80)
print()

# ==============================================================================
# Example 1: Complete Defect Study (ALL defects in one line!)
# ==============================================================================
print("Example 1: Complete Defect Study")
print("-" * 80)
print("Generates ALL defect types: vacancies + antisites")
print()

flows = RecipeBook.complete_defect_study(
    mgo,
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    charge_states=[0],
    auto_calculate_chemical_potentials=True,
    dry_run=True,
)

print(f"✅ Generated {len(flows)} defect flows:")
for i, flow in enumerate(flows, 1):
    print(f"  {i}. {flow.name}")
print()

# Uncomment to run all defects:
# for flow in flows:
#     results = run_locally(flow, create_folders=True)

# ==============================================================================
# Example 2: Vacancy Study Only
# ==============================================================================
print("Example 2: Vacancy Study Only")
print("-" * 80)
print("Generates all symmetry-unique vacancies")
print()

vacancy_flows = RecipeBook.vacancy_study(
    mgo,
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    charge_states=[0, +2],  # Neutral and charged vacancies
    auto_calculate_chemical_potentials=True,
    dry_run=True,
)

print(f"✅ Generated {len(vacancy_flows)} vacancy flows:")
for i, flow in enumerate(vacancy_flows, 1):
    print(f"  {i}. {flow.name}")
print()

# Uncomment to run: results = run_locally(vacancy_flows[0], create_folders=True)

# ==============================================================================
# Example 3: Substitution Study (Dopants)
# ==============================================================================
print("Example 3: Substitution Study (Li dopant on Mg sites)")
print("-" * 80)
print("Generates Li_Mg dopant defects")
print()

substitution_flows = RecipeBook.substitution_study(
    mgo,
    dopants="Li",  # Can also be a list: ["Li", "Na", "K"]
    species="Mg",
    charge_states=[-1, 0],  # Li+ on Mg2+ site is an acceptor
    auto_calculate_chemical_potentials=True,
    dry_run=True,
)

print(f"✅ Generated {len(substitution_flows)} substitution flows:")
for i, flow in enumerate(substitution_flows, 1):
    print(f"  {i}. {flow.name}")
print()

# Uncomment to run: results = run_locally(substitution_flows[0], create_folders=True)

# ==============================================================================
# Example 4: Antisite Study (Atom Swapping)
# ==============================================================================
print("Example 4: Antisite Study (Mg_O and O_Mg)")
print("-" * 80)
print("Generates all antisite defects (atoms swapping positions)")
print()

antisite_flows = RecipeBook.antisite_study(
    mgo,
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    charge_states=[0],
    auto_calculate_chemical_potentials=True,
    dry_run=True,
)

print(f"✅ Generated {len(antisite_flows)} antisite flows:")
for i, flow in enumerate(antisite_flows, 1):
    print(f"  {i}. {flow.name}")
print()

# Uncomment to run: results = run_locally(antisite_flows[0], create_folders=True)

# ==============================================================================
# Example 5: Interstitial Study
# ==============================================================================
print("Example 5: Interstitial Study (Li interstitials)")
print("-" * 80)
print("Generates Li interstitials at high-symmetry sites")
print()

interstitial_flows = RecipeBook.interstitial_study(
    mgo,
    species="Li",  # Can also be a list: ["Li", "H"]
    charge_states=[0, +1],
    auto_calculate_chemical_potentials=True,
    dry_run=True,
)

print(f"✅ Generated {len(interstitial_flows)} interstitial flows:")
for i, flow in enumerate(interstitial_flows, 1):
    print(f"  {i}. {flow.name}")
print()

# Uncomment to run: results = run_locally(interstitial_flows[0], create_folders=True)

# ==============================================================================
# Example 6: Complete Study with Interstitials
# ==============================================================================
print("Example 6: Complete Study with Interstitials")
print("-" * 80)
print("Vacancies + Antisites + Li interstitials")
print()

all_flows = RecipeBook.complete_defect_study(
    mgo,
    interstitial_species=["Li"],  # Add interstitials
    charge_states=[0],
    auto_calculate_chemical_potentials=True,
    dry_run=True,
)

print(f"✅ Generated {len(all_flows)} total flows:")
vacancy_count = sum(1 for f in all_flows if "vacancy" in f.name.lower())
substitution_count = sum(1 for f in all_flows if "substitution" in f.name.lower())
interstitial_count = sum(1 for f in all_flows if "interstitial" in f.name.lower())

print(f"  - Vacancies: {vacancy_count}")
print(f"  - Substitutions: {substitution_count}")
print(f"  - Interstitials: {interstitial_count}")
print()

# ==============================================================================
# Example 7: Multiple Dopants
# ==============================================================================
print("Example 7: Multiple Dopants (Li, Na, K on Mg sites)")
print("-" * 80)
print("Compare alkali metal dopants")
print()

multi_dopant_flows = RecipeBook.substitution_study(
    mgo,
    dopants=["Li", "Na", "K"],
    species="Mg",
    charge_states=[-1, 0],
    auto_calculate_chemical_potentials=True,
    dry_run=True,
)

print(f"✅ Generated {len(multi_dopant_flows)} flows:")
for dopant in ["Li", "Na", "K"]:
    count = sum(1 for f in multi_dopant_flows if dopant in f.name)
    print(f"  - {dopant}: {count} flows")
print()

# ==============================================================================
# Summary
# ==============================================================================
print("=" * 80)
print("SUMMARY: Recipe Book Defect API")
print("=" * 80)
print()
print("✅ RecipeBook.complete_defect_study() - All defects in one line")
print("✅ RecipeBook.vacancy_study() - All vacancies")
print("✅ RecipeBook.substitution_study() - Dopant substitutions")
print("✅ RecipeBook.antisite_study() - Antisite defects (atom swaps)")
print("✅ RecipeBook.interstitial_study() - Interstitial defects")
print()
print("Benefits:")
print("  • Automatic symmetry reduction (only unique sites)")
print("  • Automatic ghost atoms for SIESTA vacancies")
print("  • Automatic dielectric constant estimation")
print("  • Finite-size corrections (Lany-Zunger)")
print("  • high code reduction vs. manual approach")
print()
print("Code reduction example:")
print("  Before: ~200 lines per defect study")
print("  After:  ~10 lines for complete defect study")
print("  Reduction: high!")
print()
