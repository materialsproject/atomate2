# Mechanical Properties Recipes

Calculate mechanical properties using RecipeBook one-liners.

## Overview

Recipes for mechanical characterization:
- **Elastic Constants**: Full elastic tensor and derived properties
- **EOS and Bulk Modulus**: Energy-volume curve and bulk modulus
- **Complete Mechanical**: All mechanical properties in one workflow

## Files

- `mechanical_recipes.py` - Complete mechanical property calculations
- `README.md` - This file

## Quick Start

```python
from atomate2.siesta.recipes import RecipeBook

# Elastic constants and mechanical properties
flow = RecipeBook.elastic_constants_workflow(structure)

# EOS and bulk modulus (same calculation!)
flow = RecipeBook.eos_workflow(structure, number_of_frames=7)

# Complete mechanical properties
flow = RecipeBook.mechanical_properties(structure)
```

## Properties Calculated

### Elastic Constants Workflow
- Full elastic tensor (Cᵢⱼ)
- Bulk modulus (K)
- Shear modulus (G)
- Young's modulus (E)
- Poisson's ratio (ν)
- Mechanical stability analysis

### EOS Workflow (includes Bulk Modulus)
- Energy vs volume curve
- Bulk modulus from EOS fitting (Birch-Murnaghan or Vinet)
- Equilibrium volume
- Pressure derivative B'
- NOTE: This is the same as the old bulk_modulus_workflow()

### Mechanical Properties (Complete)
- All elastic properties
- Bulk modulus (both methods)
- Anisotropy factors
- Sound velocities
- Hardness predictions

## When to Use

- **Structural materials**: Need elastic moduli for engineering
- **Hard materials**: Diamond, carbides, nitrides
- **Mechanical stability**: Check if structure is stable
- **Pressure effects**: Compressibility studies
- **Material design**: Optimize stiffness, strength

## Computational Cost

| Workflow | Time (10 atoms, 4 cores) |
|----------|--------------------------|
| Elastic constants | 2-4 hours |
| Bulk modulus | 1-2 hours |
| Complete mechanical | 3-5 hours |

## Next Steps

- **Electronic properties**: `03_electronic_recipes/`
- **Thermal properties**: `05_thermal_recipes/`
- **Complete study**: `02_complete_workflows/`
