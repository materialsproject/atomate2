# Tutorial: Magnetic Calculations in atomate2siesta

This tutorial covers magnetic calculations in atomate2siesta, including automatic magnetic moment detection, various magnetic ordering types, and practical examples.

## Contents

1. **`01-automatic-moments/`** - Automatic detection and initialization (7 examples)
2. **`02-antiferromagnetic/`** - Antiferromagnetic (AFM) calculations (9 examples)
3. **`03-2d-materials/`** - 2D magnetic materials (9 examples: graphene, CrI₃, etc.)

## Overview

atomate2siesta automatically handles magnetic calculations by:
- Detecting magnetic elements (3d/4d transition metals, lanthanides, actinides)
- Assigning physically reasonable initial magnetic moments
- Writing the `DM.InitSpin` block in SIESTA input files
- Supporting various magnetic orderings (FM, AFM, FiM, custom)

## Quick Start

### Automatic Ferromagnetic (Default)

```python
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

# Load or create magnetic structure
fe_structure = Structure(...)  # Contains Fe atoms

# Magnetic moments set automatically!
job = RelaxMaker.fixed_cell_relaxation().make(fe_structure)
```

### Antiferromagnetic Ordering

```python
from atomate2.siesta.sets.utils import set_magnetic_ordering

# Set AFM ordering on structure
magmoms = set_magnetic_ordering(nio_structure, "AFM")
nio_structure.add_site_property("magmoms", magmoms)

# Use in workflow
job = RelaxMaker.fixed_cell_relaxation().make(nio_structure)
```

## Supported Magnetic Elements

### 3d Transition Metals (Commonly Magnetic)
- **Cr** (Z=24): 4.0 μB
- **Mn** (Z=25): 5.0 μB
- **Fe** (Z=26): 4.0 μB
- **Co** (Z=27): 3.0 μB
- **Ni** (Z=28): 2.0 μB

### 4d Transition Metals
- Mo, Tc, Ru, Rh (Z=42-45): 1.0 μB (default)

### Lanthanides
- La through Tm (Z=57-69): 1.0 μB (default)
- **Gd** (Z=64): 7.0 μB (highest)

### Actinides
- Ac through Es (Z=89-99): 1.0 μB (default)

## Magnetic Ordering Types

### 1. Ferromagnetic (FM)
All magnetic moments aligned parallel (same direction).

```
↑ ↑ ↑ ↑
```

**Example materials**: Fe, Co, Ni metals

### 2. Antiferromagnetic (AFM)
Magnetic moments alternate between up and down.

```
↑ ↓ ↑ ↓
```

**Example materials**: NiO, MnO, FeO, Cr₂O₃

### 3. Ferrimagnetic (FiM)
Like AFM but with unequal moment magnitudes on different sublattices.

```
↑↑ ↓ ↑↑ ↓
```

**Example materials**: Fe₃O₄ (magnetite), ferrites

### 4. Custom Patterns
User-defined magnetic configurations for complex structures.

## Key Functions

### `get_default_initial_magnetic_moments(structure)`
Automatically detect magnetic elements and assign default moments.

```python
from atomate2.siesta.sets.utils import get_default_initial_magnetic_moments

magmoms = get_default_initial_magnetic_moments(fe_structure)
# Returns: [4.0, 4.0] for Fe atoms
```

### `set_magnetic_ordering(structure, ordering, ...)`
Set specific magnetic ordering patterns.

```python
from atomate2.siesta.sets.utils import set_magnetic_ordering

# Ferromagnetic
magmoms_fm = set_magnetic_ordering(structure, "FM")

# Antiferromagnetic
magmoms_afm = set_magnetic_ordering(structure, "AFM")

# Ferrimagnetic
magmoms_fim = set_magnetic_ordering(structure, "FiM")

# Custom pattern
pattern = [+1, -1, +1, -1, 0, 0]
magmoms = set_magnetic_ordering(structure, "custom", afm_pattern=pattern)
```

### `get_magnetic_structure_info(structure)`
Analyze structure and get magnetic information.

```python
from atomate2.siesta.sets.utils import get_magnetic_structure_info

info = get_magnetic_structure_info(nio_structure)
print(info['magnetic_elements'])  # ['Ni']
print(info['suggested_ordering'])  # 'antiferromagnetic'
```

### `pymatgen_to_ase(..., magnetic_ordering="AFM")`
Convert structure with specific magnetic ordering.

```python
from atomate2.siesta.sets.utils import pymatgen_to_ase

# AFM ordering applied during conversion
ase_atoms = pymatgen_to_ase(nio_structure, magnetic_ordering="AFM")
```

## DM.InitSpin Block

SIESTA uses the `DM.InitSpin` block to set initial magnetic moments:

```
%block DM.InitSpin
    1  +4.000000000000000  # Atom 1: Fe, spin up
    2  -4.000000000000000  # Atom 2: Fe, spin down
%endblock DM.InitSpin
```

This block is automatically generated from the magnetic moments set on the ASE Atoms object.

## Tips and Best Practices

1. **Start with defaults**: Let the system auto-detect and assign moments
2. **For oxides**: AFM is often the ground state (NiO, MnO, etc.)
3. **For metals**: FM is typically preferred (Fe, Co, Ni)
4. **Check convergence**: Test different initial configurations
5. **Use 2D presets**: For 2D magnetic materials, use `2d_magnetic` preset

## Common Magnetic Materials

| Material | Formula | Ordering | Ground State |
|----------|---------|----------|--------------|
| Iron | Fe | FM | BCC, magnetic |
| Nickel | Ni | FM | FCC, magnetic |
| Nickel oxide | NiO | AFM (Type II) | Rock salt |
| Iron oxide | FeO | AFM | Rock salt |
| Manganese oxide | MnO | AFM | Rock salt |
| Magnetite | Fe₃O₄ | FiM | Inverse spinel |
| Chromium | Cr | AFM | BCC |
| Chromium(III) oxide | Cr₂O₃ | AFM | Corundum |
| Cobalt | Co | FM | HCP, magnetic |

## Advanced Topics

### Spin-Orbit Coupling (SOC)
For heavy elements, enable SOC in spin settings:

```python
user_params = {
    "Spin": "spin-orbit",
    "Spin.OrbitStrength": 1.0,
}
```

### Non-Collinear Magnetism
For complex magnetic textures:

```python
user_params = {
    "Spin": "non-collinear",
}
```

### Constrained Magnetic Moments
Fix total magnetization:

```python
user_params = {
    "Spin.Fix": True,
    "Spin.Total": 2.0,  # Total spin in ħ/2 units
}
```

## Troubleshooting

### Issue: SCF not converging with magnetic moments
**Solution**: Try different initial configurations (FM vs AFM) or reduce mixer weight:

```python
user_params = {
    "SCF.Mixer.Weight": 0.01,  # Reduce from default 0.05
    "SCF.Mixer.Method": "Pulay",
}
```

### Issue: Wrong magnetic ground state
**Solution**: Test multiple configurations and compare energies:

```python
# Run both FM and AFM
job_fm = make_job(structure_fm)
job_afm = make_job(structure_afm)
# Compare final energies to determine ground state
```

### Issue: Magnetic moments collapse to zero
**Solution**:
1. Ensure `Spin: polarized` is set
2. Check if material is actually magnetic
3. Try larger initial moments
4. Check pseudopotentials support magnetism

## Example Workflows

See the individual tutorial files for complete working examples of:
- Automatic magnetic moment detection
- FM, AFM, and FiM calculations
- Custom magnetic patterns
- 2D magnetic materials (CrI₃, VSe₂)
- Magnetic oxide calculations (NiO, Fe₂O₃)

## References

1. SIESTA Manual: Spin polarization section
2. [DFT magnetism tutorial](https://docs.materialsproject.org/)
3. Pymatgen magnetic ordering analysis

## Next Steps

After completing this tutorial:
1. Try the Recipe Book for quick magnetic workflows
2. Explore DFT+U for strongly correlated systems (Tutorial 08)
3. Learn about phonon calculations in magnetic materials (Tutorial 15)

---

**Tutorial version**: 1.0
**Last updated**: November 5, 2025
**Requires**: atomate2siesta v1.0.0+
