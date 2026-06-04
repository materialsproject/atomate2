# Thermal Properties Recipes

Calculate vibrational and thermal properties using RecipeBook one-liners.

## Overview

Recipes for thermal and vibrational characterization:
- **Phonons**: Phonon band structure and DOS
- **QHA**: Quasi-harmonic approximation for thermal expansion
- **Grüneisen**: Grüneisen parameters and mode analysis
- **Complete Thermal**: All thermal properties combined

## Files

- `phonon_recipes.py` - Phonon calculations (HIGH priority)
- `qha_recipes.py` - Quasi-harmonic approximation (MEDIUM priority)
- `gruneisen_recipes.py` - Grüneisen parameters (MEDIUM priority)
- `thermal_properties_all.py` - Complete thermal workflow (HIGH priority)
- `README.md` - This file

## Quick Start

```python
from atomate2.siesta.recipes import RecipeBook

# Phonon properties
flow = RecipeBook.phonon_workflow(structure)

# Thermal expansion via QHA
flow = RecipeBook.qha_workflow(structure)

# Grüneisen parameters
flow = RecipeBook.gruneisen_workflow(structure)

# All thermal properties
flow = RecipeBook.thermal_properties(structure)
```

## Properties Calculated

### Phonon Workflow
- Phonon band structure
- Phonon density of states
- Thermodynamic properties (Cv, S, F)
- Zero-point energy
- Automatic plots

### QHA Workflow
- Thermal expansion α(T)
- Temperature-dependent bulk modulus
- Heat capacity at constant pressure
- Gibbs free energy G(T,P)

### Grüneisen Workflow
- Mode Grüneisen parameters
- Average Grüneisen parameter
- Thermal expansion from γ
- Volume-dependent phonons

### Complete Thermal Properties
- All phonon properties
- QHA thermal expansion
- Grüneisen analysis
- Comprehensive thermal characterization

## When to Use

- **Thermal stability**: Check for imaginary modes
- **Thermal expansion**: Temperature-dependent lattice
- **Heat capacity**: Cv and Cp calculations
- **Phase transitions**: Temperature-induced changes
- **Materials design**: Thermal management applications

## Computational Cost

| Workflow | Time (10 atoms, 4 cores) | Supercell Size |
|----------|--------------------------|----------------|
| Phonons | 4-8 hours | 2×2×2 |
| QHA | 12-24 hours | 2×2×2 + volumes |
| Grüneisen | 16-32 hours | Multiple volumes |
| Complete thermal | 20-36 hours | All above |

**Note**: Computational cost scales strongly with supercell size!

## Best Practices

1. **Start with small supercell** (2×2×2) for testing
2. **Check convergence** with larger supercells (3×3×3)
3. **Use tight parameters** for force calculations
4. **Enable custodian** for automatic error handling
5. **Monitor disk space** (phonon calculations produce many files)

## Next Steps

- **Phonon details**: `tutorials/05-vibrational-properties/`
- **Electronic properties**: `03_electronic_recipes/`
- **Complete study**: `02_complete_workflows/`
