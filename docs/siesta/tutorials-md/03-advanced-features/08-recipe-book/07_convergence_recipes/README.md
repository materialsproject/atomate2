# Convergence Testing Recipes

Automated convergence testing using RecipeBook one-liners.

## Overview

Recipes for parameter convergence studies:
- **K-Points Convergence**: Automatic k-mesh convergence testing
- **Basis Convergence**: Basis parameter optimization
- **Complete Convergence**: Full convergence suite

## Files

- `kpoints_convergence_recipes.py` - k-point convergence (HIGH priority)
- `basis_convergence_recipes.py` - Basis parameter convergence (HIGH priority)
- `complete_convergence_recipes.py` - Full convergence study (MEDIUM priority)
- `README.md` - This file

## Quick Start

```python
from atomate2.siesta.recipes import RecipeBook

# k-point convergence
flow = RecipeBook.kpoints_convergence(structure)

# Basis parameter convergence
flow = RecipeBook.basis_convergence(structure)

# Complete convergence study
flow = RecipeBook.complete_convergence(structure)
```

## Properties Tested

### K-Points Convergence
- Energy vs k-mesh density
- Automatic convergence criterion
- Recommended k-mesh
- Convergence plot

### Basis Convergence
- PAO.EnergyShift convergence
- PAO.SplitNorm optimization
- PAO.BasisSize comparison
- Accuracy vs cost tradeoff

### Complete Convergence
- k-point convergence
- Mesh cutoff convergence
- Basis parameter convergence
- Production-ready parameters

## When to Use

- **Before production calculations**: Always test convergence first
- **New material types**: Different materials need different parameters
- **Publication work**: Document converged parameters
- **Method development**: Understand parameter effects
- **Troubleshooting**: Check if parameters are adequate

## Convergence Criteria

### Energy Convergence
- Target: < 0.01 eV/atom (tight)
- Acceptable: < 0.05 eV/atom (standard)
- Screening: < 0.1 eV/atom (loose)

### Force Convergence
- Target: < 0.01 eV/Å (tight)
- Acceptable: < 0.05 eV/Å (standard)

### Stress Convergence
- Target: < 0.1 GPa (tight)
- Acceptable: < 0.5 GPa (standard)

## Typical Converged Parameters

### Metals
```
k-points: 8×8×8 or higher
Mesh.Cutoff: 300-400 Ry
PAO.BasisSize: DZP
ElectronicTemperature: 100-500 K
```

### Semiconductors
```
k-points: 6×6×6 to 8×8×8
Mesh.Cutoff: 300-400 Ry
PAO.BasisSize: DZP or TZP
```

### Insulators
```
k-points: 4×4×4 to 6×6×6
Mesh.Cutoff: 250-350 Ry
PAO.BasisSize: DZP
```

### Molecules
```
k-points: 1×1×1 (Gamma point)
Mesh.Cutoff: 300-400 Ry
PAO.BasisSize: DZP or TZP
```

## Computational Cost

| Workflow | Time (10 atoms, 4 cores) | Tests Performed |
|----------|--------------------------|-----------------|
| k-points | 1-2 hours | 5-7 k-meshes |
| Basis | 2-4 hours | 10-15 parameters |
| Complete | 4-8 hours | All tests |

## Best Practices

1. **Always test convergence** before production
2. **Test one parameter at a time**
3. **Start with coarse, refine progressively**
4. **Document converged parameters**
5. **Use same parameters for series**
6. **Retest for different properties**
7. **Plot convergence curves**

## Next Steps

- **Apply converged parameters**: `02_complete_workflows/`
- **Production calculations**: `08_combined_recipes/`
- **Manual convergence**: `tutorials/02-convergence/`
