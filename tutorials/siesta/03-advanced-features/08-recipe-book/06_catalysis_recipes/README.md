# Catalysis and Surface Chemistry Recipes

Calculate surface and adsorption properties using RecipeBook one-liners.

## Overview

Recipes for catalysis and surface chemistry:
- **Surface Energy**: Multi-surface energy calculations
- **Adsorption**: Adsorption site scanning and energies
- **Catalysis Study**: Complete catalytic characterization

## Files

- `surface_energy_recipes.py` - Surface energy calculations (HIGH priority)
- `adsorption_recipes.py` - Adsorption site screening (HIGH priority)
- `catalysis_study_recipes.py` - Complete catalysis workflow (MEDIUM priority)
- `README.md` - This file

## Quick Start

```python
from atomate2.siesta.recipes import RecipeBook

# Surface energies for multiple facets
flow = RecipeBook.surface_energy_workflow(structure, miller_indices=[(1,0,0), (1,1,0), (1,1,1)])

# Adsorption site scanning
flow = RecipeBook.adsorption_scanning_workflow(structure, adsorbate="O")

# Complete catalysis study
flow = RecipeBook.catalysis_study(structure, adsorbates=["H", "O", "OH"])
```

## Properties Calculated

### Surface Energy Workflow
- Surface energies for multiple facets
- Surface stability plot
- Wulff construction
- Preferred surface terminations

### Adsorption Workflow
- Adsorption energies at multiple sites
- Site preference ranking
- Coverage effects
- Adsorption geometry

### Complete Catalysis Study
- Multiple surface facets
- Adsorption on each surface
- Reaction barriers (if NEB included)
- Activity descriptors

## When to Use

- **Heterogeneous catalysis**: Surface reactions
- **Electrocatalysis**: ORR, OER, HER, CO2RR
- **Surface science**: Adsorption studies
- **Materials screening**: Catalyst discovery
- **Mechanism studies**: Reaction pathways

## Applications

### Water Splitting (OER/HER)
```python
flow = RecipeBook.catalysis_study(
    structure,
    adsorbates=["H", "O", "OH", "OOH"],  # OER intermediates
    miller_indices=[(1,0,0), (1,1,1)],
)
```

### CO2 Reduction
```python
flow = RecipeBook.catalysis_study(
    structure,
    adsorbates=["CO2", "CO", "COOH", "CHO"],
    miller_indices=[(1,1,1)],
)
```

### Hydrogenation Reactions
```python
flow = RecipeBook.adsorption_scanning_workflow(
    structure,
    adsorbate="H",
    coverage_range=[0.25, 0.50, 1.0],
)
```

## Computational Cost

| Workflow | Time (per surface, 4 cores) |
|----------|----------------------------|
| Surface energy | 2-4 hours |
| Adsorption (10 sites) | 4-8 hours |
| Complete catalysis | 12-24 hours |

**Note**: Cost scales with number of surfaces and adsorbates!

## Best Practices

1. **Start with low-index surfaces** (100, 110, 111)
2. **Use appropriate slab thickness** (≥ 4-5 atomic layers)
3. **Include vacuum** (≥ 15 Å)
4. **Test convergence** with k-points and layers
5. **Consider dispersion** for molecular adsorbates
6. **Use spin polarization** for magnetic surfaces

## Next Steps

- **Surface details**: `tutorials/06-surfaces-and-adsorption/`
- **NEB for barriers**: `tutorials/03-advanced-workflows/04-neb/`
- **Electronic properties**: `03_electronic_recipes/`
