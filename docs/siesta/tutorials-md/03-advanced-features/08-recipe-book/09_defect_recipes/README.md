# Defect Recipes

This directory contains tutorials for using the Recipe Book API for point defect calculations.

## Overview

The defect recipes provide the simplest way to perform defect calculations in atomate2siesta. With one-liners, you can generate complete defect studies including:

- All symmetry-unique vacancies
- Antisite defects (atom swapping)
- Substitutional dopants
- Interstitial defects

## Key Features

- **Automatic symmetry reduction**: Only unique sites are generated
- **Ghost atoms**: SIESTA-specific ghost atoms for vacancies (automatic)
- **Finite-size corrections**: Lany-Zunger corrections applied automatically
- **Dielectric constant estimation**: Auto-estimated from material type
- **high code reduction**: ~200 lines → ~10 lines per study

## Available Recipes

### RecipeBook.complete_defect_study()
Generate ALL defect types in one line:
```python
flows = RecipeBook.complete_defect_study(
    structure,
    supercell_matrix=[[2,0,0], [0,2,0], [0,0,2]],
    charge_states=[0],
    interstitial_species=["Li"],  # Optional
)
```

### RecipeBook.vacancy_study()
All symmetry-unique vacancies:
```python
flows = RecipeBook.vacancy_study(
    structure,
    charge_states=[0, +2],
)
```

### RecipeBook.substitution_study()
Dopant substitutions:
```python
flows = RecipeBook.substitution_study(
    structure,
    dopants=["Li", "Na", "K"],
    species="Mg",
    charge_states=[-1, 0],
)
```

### RecipeBook.antisite_study()
Antisite defects (atom swaps):
```python
flows = RecipeBook.antisite_study(
    structure,
    charge_states=[0],
)
```

### RecipeBook.interstitial_study()
Interstitial defects at high-symmetry sites:
```python
flows = RecipeBook.interstitial_study(
    structure,
    species=["Li", "H"],
    charge_states=[0, +1],
)
```

## Tutorial Files

### defect_recipes.py
Complete demonstration of all defect recipes with examples for:
1. Complete defect study (all types)
2. Vacancy study only
3. Substitution study (dopants)
4. Antisite study
5. Interstitial study
6. Multiple dopants comparison

## Running the Tutorial

```bash
# Preview all examples (dry_run mode enabled in tutorial)
python defect_recipes.py

# To run actual calculations, edit the file and uncomment the run_locally() calls
```

## Code Comparison

**Before** (Manual approach, ~200 lines):
```python
# Create defect structure manually
defect_structure = structure.copy()
defect_structure.remove_sites([0])

# Create supercell manually
supercell = defect_structure * (2, 2, 2)

# Set up DefectFlowMaker
maker = DefectFlowMaker(...)
flow = maker.make(
    defect_structure=supercell,
    host_structure=structure,
    defect_site=[0, 0, 0],
    defect_species="vacancy",
    charge_state=0,
    epsilon_static=9.8,
)
# ... repeat for each defect type and charge state (100+ lines)
```

**After** (Recipe Book, ~10 lines):
```python
# Generate all defects in ONE LINE!
flows = RecipeBook.complete_defect_study(
    structure,
    charge_states=[0, +2],
)
# Done! All vacancies, antisites, and interstitials generated automatically
```

**Result**: high code reduction, zero manual structure manipulation

## See Also

- Lower-level API: `tutorials/02-workflows/05-defects/03-DefectFlowMaker/`
- Defect flow documentation: `docs/source/defect-workflows.rst`
- Recipe Book overview: `tutorials/03-advanced-features/08-recipe-book/README.md`
