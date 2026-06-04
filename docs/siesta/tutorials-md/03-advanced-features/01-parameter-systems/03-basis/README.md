# Tutorial 07: Basis Set Customization

This tutorial series demonstrates how to customize basis sets in SIESTA calculations using atomate2siesta. Basis sets are crucial for balancing accuracy and computational cost.

## Tutorial Overview

| File | Description | Complexity |
|------|-------------|------------|
| `01_global_basis_size.py` | Set global basis size (SZ, DZ, DZP, TZP) | ⭐ Beginner |
| `02_single_atom_override.py` | Per-species basis with %block PAO.BasisSizes | ⭐⭐ Beginner |
| **`03_1_per_atom_direct.py`** | **🆕 Per-atom basis (direct specification)** | ⭐⭐ Intermediate |
| **`03_2_per_atom_grouped.py`** | **🆕 Per-atom basis (grouped/layer-based)** | ⭐⭐ Intermediate |
| `04_custom_pao_basis_simple.py` | Full custom basis specification with %block PAO.Basis | ⭐⭐⭐ Advanced |
| `05_custom_pao_basis_with_polarization.py` | Custom basis + polarization orbitals | ⭐⭐⭐ Advanced |
| `06_multispecies_custom_basis.py` | Custom basis for multiple species (Mg+O) | ⭐⭐⭐⭐ Advanced |
| `07_combining_blocks.py` | Combine PAO.BasisSize with %block PAO.Basis | ⭐⭐⭐⭐ Expert |
| **`08_species_variants_surface.py`** | **🆕 Species variants for surface calculations (dict format)** | ⭐⭐ Intermediate |
| **`09_species_variants_adsorption.py`** | **🆕 Species variants for adsorption studies** | ⭐⭐⭐ Intermediate-Advanced |
| **`10_species_variants_bsse_ghost.py`** | **🆕 Species variants for BSSE corrections (ghost atoms)** | ⭐⭐⭐⭐ Advanced |
| **`11_pao_basis_helpers.py`** | **🆕 PAO.Basis helper functions (programmatic generation)** | ⭐⭐⭐ Advanced |
| **`12_reading_ghost_atoms.py`** | **🆕 Reading structures with ghost atoms (CIF/FDF)** | ⭐ Beginner |

## Quick Reference

### Standard Basis Sizes

```python
"PAO.BasisSize": "SZ"   # Single-Zeta (minimal, fast)
"PAO.BasisSize": "DZ"   # Double-Zeta (better)
"PAO.BasisSize": "SZP"  # Single-Zeta + Polarization
"PAO.BasisSize": "DZP"  # Double-Zeta + Polarization (recommended)
"PAO.BasisSize": "TZP"  # Triple-Zeta + Polarization (high accuracy)
```

### Per-Species Basis Override

```python
user_params = {
    "%block PAO.BasisSizes": ["Mg DZP", "O TZP"],  # Per-species (element type)
    "a2s_kpts": [6, 6, 6],
    "Mesh.Cutoff": "350 Ry",
}
```

Note: This is per-SPECIES (element type), not per-atom. All Mg atoms get DZP, all O atoms get TZP.

### Per-Atom Basis (🆕 NEW in v1.0.0)

The **per-atom basis helpers** enable atom-level precision control:

```python
from atomate2.siesta.sets.utils import apply_per_atom_basis

# Method 1: Direct per-atom specification (1-indexed like SIESTA)
per_atom_basis = {
    1: "TZP",   # Atom 1 (surface)
    2: "TZP",   # Atom 2 (subsurface)
    3: "DZP",   # Atom 3 (bulk)
    # Rest use fallback (DZ)
}

species_labels, pao_basissizes = apply_per_atom_basis(
    structure, per_atom_basis, fallback_basis="DZ"
)
structure.add_site_property("species_label", species_labels)

# Method 2: Grouped specification (layer-based)
from atomate2.siesta.sets.utils import create_per_atom_basis_dict

atom_groups = {
    "surface": ([1, 2, 3], "TZP"),       # Atoms 1-3
    "bulk": ([4, 5, 6], "DZ"),           # Atoms 4-6
}

species_labels, pao_basissizes = create_per_atom_basis_dict(
    structure, atom_groups
)

# Use with RelaxMaker
maker = RelaxMaker(user_params={'%block PAO.BasisSizes': pao_basissizes})
```

**Use Cases**:
- **Surface slabs**: Different basis for surface vs bulk atoms (even same element)
- **Defects**: High accuracy around defect site
- **Dopants**: Special treatment for specific doped atoms
- **Layer-based systems**: Automatic grouping by z-coordinate

See tutorials 03.1 and 03.2 for detailed examples!

### Species Variants (🆕 NEW in v1.0.0)

The **dict format** enables species variants like O_surface, O_bulk, O_ghost:

```python
user_params = {
    "%block PAO.BasisSizes": {
        "Ti_bulk": "DZP",       # Bulk atoms - standard accuracy
        "Ti_surface": "TZP",    # Surface atoms - high accuracy
        "O_bulk": "DZ",         # Bulk oxygen
        "O_surface": "TZP",     # Surface oxygen
        "H_ghost": "DZP",       # Ghost atoms for BSSE
    },
    "a2s_kpts": [4, 4, 1],
    "Mesh.Cutoff": "300 Ry",
}
```

**Use Cases**:
- **Surface calculations**: Different basis for surface vs bulk atoms
- **Adsorption studies**: High accuracy for adsorbate + interface, standard for substrate
- **BSSE corrections**: Ghost atoms (Z < 0) for counterpoise method

See tutorials 08, 09, 10 for detailed examples!

### PAO.Basis Helper Functions (🆕 NEW in v1.0.0)

The **PAO.Basis builder** enables programmatic generation of custom orbital blocks:

```python
from atomate2.siesta.sets.utils.basis_builder import create_pao_basis

# Define custom basis specification
basis_spec = {
    "Ti": {
        "shells": [
            {"n": 4, "l": 0, "nzeta": 2, "rc": [6.0, 0.0]},  # 4s: 2 zeta
            {
                "n": 3,
                "l": 2,
                "nzeta": 2,
                "rc": [7.0, 0.0],
                "polarization": True,  # Add polarization orbital
            },
        ]
    },
    "O_surface": {  # Works with species variants!
        "shells": [
            {"n": 2, "l": 0, "nzeta": 2, "rc": [6.0, 0.0]},
            {
                "n": 2,
                "l": 1,
                "nzeta": 2,
                "rc": [7.0, 0.0],
                "polarization": True,
                "split_norm": 0.25,  # Advanced PAO flags supported
            },
        ]
    },
}

# Generate PAO.Basis block (returns list format)
pao_basis = create_pao_basis(basis_spec)

# Use with RelaxMaker
maker = RelaxMaker(user_params={'%block PAO.Basis': pao_basis})
```

**Key Features**:
- Programmatic generation (no manual FDF string formatting)
- Validation: nzeta vs rc length, l range, split_norm
- All SIESTA PAO flags: polarization, split_norm, soft_conf, charge_conf, etc.
- Species variants support: O_surface, O_bulk, O_ghost, etc.
- Returns list format (ready for user_params)

**Use Cases**:
- Custom cutoff radii optimization
- Species variants with different orbital configurations
- Advanced basis set development
- Polarization orbital control
- Combine with per-atom helpers for ultimate control

See tutorial 11 for detailed examples!

### Custom Basis Block

```python
custom_basis_block = """Si  3
  n=3  0  2  E  50.0  4.5    # 3s: 2 zeta
    5.0  3.5
  n=3  1  2  E  50.0  5.0    # 3p: 2 zeta
    5.5  4.0
  n=3  2  1  P  1             # 3d: polarization
    5.0
"""

user_params = {
    "PAO.BasisSize": "DZP",  # Fallback for species not in block
    "%block PAO.Basis": custom_basis_block.strip().split('\n'),  # Must be list!
    "a2s_kpts": [6, 6, 6],
    "Mesh.Cutoff": "300 Ry",
}
```

**Important**: `%block PAO.Basis` must be a **list of strings** (split by newline), not a single string!

## PAO.Basis Format Explained

```
Species  NumberOfShells
  n=N  l  Nzeta  [Type]  [Params]
    rc1  rc2  ...  (cutoff radii in Bohr)
```

Where:
- `Species`: Element symbol (Si, O, etc.)
- `NumberOfShells`: Number of angular momentum shells
- `n=N`: Principal quantum number
- `l`: Angular momentum (0=s, 1=p, 2=d, 3=f)
- `Nzeta`: Number of zeta functions (1=SZ, 2=DZ, 3=TZ)
- `Type`: Optional flags:
  - `E`: Energy shift method (followed by energy in meV, split norm)
  - `P`: Polarization orbital (followed by polarization index)
- `rc1, rc2, ...`: Cutoff radii in Bohr (one per zeta)

## Common Scenarios

### 1. Production Calculations
```python
"PAO.BasisSize": "DZP"  # Good accuracy-cost balance
```

### 2. High-Accuracy Benchmarks
```python
"PAO.BasisSize": "TZP"  # Use with caution (expensive!)
```

### 3. Multi-Species System (e.g., MgO)
```python
user_params = {
    "%block PAO.BasisSizes": ["Mg DZP", "O TZP"],  # Per-species
    "a2s_kpts": [6, 6, 6],
    "Mesh.Cutoff": "350 Ry",
}
```

### 4. Custom Basis for One Species
```python
custom_o_basis = """O  3
  n=2  0  2  E  50.0  3.5
    4.0  2.5
  n=2  1  2  E  50.0  4.0
    4.5  3.0
  n=3  2  1  P  1
    4.0
"""

user_params = {
    "PAO.BasisSize": "TZP",  # Mg uses standard TZP
    "%block PAO.Basis": custom_o_basis.strip().split('\n'),  # O uses custom
    "a2s_kpts": [6, 6, 6],
}
```

Note: When both are present, PAO.Basis takes priority for its species, PAO.BasisSize is fallback for others.

## Running the Tutorials

Each tutorial is standalone:

```bash
cd /path/to/tutorials/01-basics/07-basis-set-customization
python 01_global_basis_size.py
python 02_single_atom_override.py
# ... etc
```

All tutorials use `dry_run=True` so they generate FDF files without running SIESTA. Check the generated `siesta.fdf` files in the `dry_run_output` directories.

## Important Notes

⚠️ **Block Format**:
- `%block PAO.Basis` must be a **list of strings** (one per line)
- Use `.strip().split('\n')` to convert multi-line string to list
- Direct string assignment will NOT work!

⚠️ **Basis Convergence**:
- Always test convergence with respect to basis size
- Total energy differences should converge to desired accuracy
- Larger basis ≠ always better (diminishing returns)

⚠️ **Computational Cost**:
- SZ/DZ: ~1x (baseline)
- SZP/DZP: ~2-3x slower than SZ/DZ
- TZP: ~5-10x slower than DZP
- Custom basis: Depends on number of orbitals

## Best Practices

1. **Start Simple**: Use standard basis sizes (DZP recommended)
2. **Test Convergence**: Verify results with larger basis (TZP)
3. **Understand Priority**: %block PAO.Basis > %block PAO.BasisSizes > PAO.BasisSize
4. **Per-Species vs Per-Atom**: Current support is per-SPECIES (element type), not per-atom
5. **List Format**: Always use `.split('\n')` for blocks (they must be lists!)
6. **Coexistence**: PAO.BasisSize can coexist with %block PAO.Basis (fallback mechanism)
7. **Document Choices**: Note basis decisions in your workflow
8. **Validate**: Compare with established basis sets when possible

## Further Reading

- SIESTA Manual: Section on PAO Basis Sets
- `atomate2siesta-pseudos basis` command: Generate basis blocks from pseudopotentials
- Tutorial 02-convergence: Basis convergence studies

## See Also

- `../02-BandStructureMaker/`: General parameter customization
- `../../../02-workflows/01-convergence/02-basis-convergence/`: Systematic basis convergence
- SIESTA Documentation: [https://siesta-project.org/](https://siesta-project.org/)
