# Electronic Properties Recipes

Calculate electronic structure properties using RecipeBook one-liners.

## Overview

This section covers recipes for electronic properties:
- **Band Structure**: Electronic band structure along high-symmetry paths
- **Density of States (DOS)**: Total and projected DOS
- **Optical Properties**: Absorption, dielectric function, reflectivity
- **Complete Electronic Properties**: Combined workflow

## Files

- `band_structure_recipes.py` ✅ - Band structure calculations
- `dos_recipes.py` - Density of states calculations
- `optical_properties_recipes.py` - Optical absorption and dielectric properties
- `electronic_properties_all.py` - Complete electronic characterization

## When to Use

### Band Structure Recipe
Use when you need to:
- Identify band gap (direct vs indirect)
- Visualize valence and conduction bands
- Determine VBM and CBM locations
- Calculate effective masses
- Understand electronic character (metal/semiconductor/insulator)

### DOS Recipe
Use when you need to:
- Detailed density of states analysis
- Projected DOS by atom or orbital
- Understand orbital contributions
- Analyze hybridization
- Check electron count and Fermi level

### Optical Properties Recipe
Use when you need to:
- Optical absorption spectrum
- Dielectric function (ε₁, ε₂)
- Refractive index
- Reflectivity
- Applications: solar cells, optoelectronics

### Complete Electronic Properties
Use when you need:
- Publication-quality electronic structure
- All electronic properties together
- Comprehensive characterization
- Comparison across materials

## Quick Start

### Band Structure
```python
from atomate2.siesta.recipes import RecipeBook
from pymatgen.core import Structure

structure = Structure.from_file("POSCAR")
flow = RecipeBook.band_structure_workflow(structure)
```

### DOS
```python
flow = RecipeBook.dos_workflow(structure)
```

### Optical Properties
```python
flow = RecipeBook.optical_properties(structure)
```

### All Electronic Properties
```python
flow = RecipeBook.electronic_properties(structure)
```

## Key Features

- **Automatic k-path generation**: Uses material-specific high-symmetry paths
- **Publication-quality plots**: Automatically generated figures
- **Material-specific presets**: Optimized for metals, semiconductors, insulators
- **Customizable parameters**: Override any setting
- **Database integration**: Results automatically stored

## Properties Calculated

### Band Structure Workflow
- Electronic bands along k-path
- Band gap (value, type, locations)
- VBM and CBM energies and positions
- Fermi level
- Band structure plot

### DOS Workflow
- Total density of states
- Projected DOS (by atom)
- Projected DOS (by orbital: s, p, d, f)
- Integrated DOS
- DOS plot

### Optical Properties Workflow
- Absorption coefficient α(ω)
- Dielectric function ε(ω) = ε₁ + iε₂
- Refractive index n(ω)
- Reflectivity R(ω)
- Optical gap
- Optical property plots

### Complete Electronic Properties
All of the above plus:
- Charge density analysis
- Electronic structure summary
- Comprehensive plots

## Material-Specific Considerations

### Metals
```python
flow = RecipeBook.band_structure_workflow(
    structure,
    tier_preset="metals_standard",  # Includes smearing
)
```
- Use electronic temperature (smearing)
- No band gap
- Check Fermi surface topology

### Semiconductors
```python
flow = RecipeBook.band_structure_workflow(
    structure,
    tier_preset="semiconductors_standard",
)
```
- Identify gap type (direct/indirect)
- Calculate VBM/CBM effective masses
- Consider hybrid functionals for accurate gaps

### Insulators
```python
flow = RecipeBook.band_structure_workflow(
    structure,
    tier_preset="oxides_standard",  # For oxides
)
```
- Large band gaps (> 3 eV)
- May need hybrid functionals
- Check for localized vs delocalized states

## Accuracy Considerations

### Standard Accuracy (Default)
- k-mesh: 4×4×4 (SCF)
- Basis: DZ or DZP
- Cutoff: 250-300 Ry
- Good for screening

### High Accuracy (Publications)
```python
flow = RecipeBook.band_structure_workflow(
    structure,
    relax_maker_kwargs={
        "user_params": {
            "kpts": [8, 8, 8],
            "Mesh.Cutoff": "400 Ry",
            "PAO.BasisSize": "DZP",
        }
    },
    line_density=30,  # Very smooth bands
)
```

## Common Customizations

### Custom k-Path
```python
flow = RecipeBook.band_structure_workflow(
    structure,
    kpath_scheme="setyawan_curtarolo",
    line_density=25,
)
```

### Energy Range for DOS
```python
flow = RecipeBook.dos_workflow(
    structure,
    dos_kwargs={
        "energy_range": [-10, 10],  # eV relative to Fermi
        "energy_step": 0.01,
    }
)
```

### Optical Spectrum Range
```python
flow = RecipeBook.optical_properties(
    structure,
    energy_range=[0, 10],  # eV
    smearing=0.1,  # eV
)
```

## Computational Cost

Typical timing for 10-atom structure on 4 cores:

| Workflow | Time | Relative Cost |
|----------|------|---------------|
| Band structure | 10-15 min | 1× |
| DOS | 12-18 min | 1.2× |
| Optical | 20-30 min | 2× |
| Complete electronic | 45-60 min | 4× |

## Troubleshooting

### Problem: Band gap too small
- Increase mesh cutoff
- Use better basis set (DZP or TZP)
- Consider hybrid functionals
- Check structure is relaxed

### Problem: Discontinuous bands
- Increase line_density
- Use denser SCF k-mesh
- Check k-path is appropriate

### Problem: DOS shows gap but bands don't
- k-path may not sample gap location
- Use automatic k-path generation
- Verify gap is not at gamma point

### Problem: Optical spectrum noisy
- Increase k-mesh density
- Add smearing
- Check convergence

## Examples

See individual tutorial files for:
- Basic usage for each recipe
- Material-specific examples
- High-accuracy calculations
- Comparison across materials
- Troubleshooting scenarios

## Best Practices

1. **Start with band structure** to understand electronic character
2. **Add DOS** for detailed orbital analysis
3. **Include optical** if relevant to application
4. **Use complete workflow** for comprehensive study
5. **Test convergence** for k-mesh and cutoff
6. **Compare with experiments** for validation
7. **Use tier presets** for material-specific optimization

## Next Steps

- **Mechanical properties**: See `04_mechanical_recipes/`
- **Thermal properties**: See `05_thermal_recipes/`
- **Complete study**: See `02_complete_workflows/`
- **Advanced features**: See `tutorials/07-advanced-features/`

## Further Reading

- RecipeBook API: `docs/source/api/recipes.rst`
- Electronic structure theory: `docs/source/theory/electronic_structure.rst`
- Convergence testing: `07_convergence_recipes/`
- Manual approach: `tutorials/01-basics/03-band-structure/`
