# Recipe Book Customization Examples

This folder contains **minimal working examples** demonstrating how to customize Recipe Book workflows.

## Files in This Folder

- **`00_FULL_TUTORIAL.py`** - Complete tutorial with all examples (reference)
- **`01_user_params.py`** - Method 1: Direct parameter customization ⭐ START HERE
- **`02_tier_levels.py`** - Method 2: Computational rigor levels
- **`03_presets.py`** - Method 3: Material-specific presets
- **`04_combining_methods.py`** - Method 4: Preset + user_params (RECOMMENDED!)
- **`05_magnetic_dftu.py`** - Advanced: Magnetic + DFT+U
- **`06_vdw_2d_materials.py`** - Advanced: 2D materials + vdW
- **`07_recipe_specific_params.py`** - Workflow-specific parameters
- **`README.md`** - This file

## Quick Start

Run any example directly:

```bash
cd tutorials/03-advanced-features/08-recipe-book/10_customizing_recipes/
python 01_user_params.py  # Start here!
```

All examples use `dry_run=True` to generate input files without running calculations.

## Examples Overview

### Method 1: Direct Parameter Customization
**`01_user_params.py`** - Basic FDF parameter customization
- Most common approach
- Pass parameters directly in `user_params` dictionary
- **IMPORTANT**: Use `auto_params=False` to disable automatic parameter detection
- Example: Custom basis, cutoff, k-points, convergence

**Key takeaway**: Use `auto_params=False` + `user_params={...}` for full control

---

### Method 2: Tier Levels
**`02_tier_levels.py`** - Computational rigor levels
- 5 tier levels: `basic_dirty`, `basic`, `intermediate`, `advanced`, `expert`
- Controls overall accuracy vs speed
- Example: Publication-quality with `tier='advanced'`

**Key takeaway**: Use `tier` parameter to set computational level

---

### Method 3: Material-Specific Presets
**`03_presets.py`** - Optimized parameter sets
- material-specific presets for different material types
- Examples: `surface_metal`, `2d_vdw`, `phonon_high_accuracy`
- View all: `atomate2siesta-presets list`

**Key takeaway**: Use `preset` for material-optimized defaults

---

### Method 4: Combining Methods (RECOMMENDED!)
**`04_combining_methods.py`** - Preset + user_params
- Start with preset, override specific parameters
- Best of both worlds: optimization + customization
- Priority: tier → preset → user_params (highest)

**Key takeaway**: This is the **recommended approach** for most cases

---

### Advanced Examples

**`05_magnetic_dftu.py`** - Spin-polarized and DFT+U
- Magnetic calculations with `Spin='polarized'`
- DFT+U for correlated systems (NiO, CuO, etc.)
- Correct SIESTA syntax for DFTU.Proj block

**Key takeaway**: Use `DFTU.ProjectorGenerationMethod` + `%block DFTU.Proj`

---

**`06_vdw_2d_materials.py`** - van der Waals functionals
- 2D materials (graphene, MoS2, h-BN)
- vdW functionals: DRSLL, C09, BH, KBM
- Proper k-mesh for 2D ([12,12,1])

**Key takeaway**: Use `vdw='DRSLL'` + `preset='2d_vdw'`

---

**`07_recipe_specific_params.py`** - Workflow parameters
- Phonon: `supercell_matrix`, `min_length`
- Surface: `miller_indices`, `slab_layers`, `vacuum`
- EOS: `number_of_frames`
- Separate from FDF parameters!

**Key takeaway**: Workflow params vs FDF params are different

---

## Understanding auto_params

By default, Recipe Book workflows use `auto_params=True`, which automatically analyzes your structure and sets optimal parameters (k-points, cutoff, basis). This is convenient but can override your explicit `user_params`.

**When to use `auto_params=False`:**
- You want complete control over all parameters
- You're setting specific k-points, cutoff, or basis values
- You're following a specific computational protocol

**When to use `auto_params=True` (default):**
- You want smart defaults based on your material type
- You're combining automatic + custom parameters
- You trust the MaterialAnalyzer recommendations

**Example:**
```python
# Full control (recommended for tutorials)
flow = RecipeBook.eos_workflow(
    structure,
    auto_params=False,  # Disable automatic detection
    user_params={'PAO.BasisSize': 'TZP', 'a2s_kpts': [8,8,8]}
)

# Smart defaults + custom overrides
flow = RecipeBook.eos_workflow(
    structure,
    auto_params=True,  # Auto-detect material properties
    user_params={'Spin': 'polarized'}  # Add magnetic calculation
)
```

## Common Patterns

### Pattern 1: Quick Testing
```python
flow = RecipeBook.phonon_workflow(
    structure,
    tier='basic',  # Fast parameters
    dry_run=True
)
```

### Pattern 2: Publication Quality
```python
flow = RecipeBook.band_structure_workflow(
    structure,
    tier='advanced',           # High accuracy
    user_params={
        'PAO.BasisSize': 'TZP',
        'Mesh.Cutoff': '500 Ry',
    }
)
```

### Pattern 3: Material-Specific
```python
flow = RecipeBook.surface_energy_workflow(
    bulk,
    preset='surface_metal',    # Optimized for metals
    user_params={
        'a2s_kpts': [8, 8, 1],  # Custom k-mesh
    }
)
```

### Pattern 4: Complex System (DFT+U + Magnetic)
```python
flow = RecipeBook.elastic_constants_workflow(
    nio,
    preset='magnetic_correlated',
    user_params={
        'Spin': 'polarized',
        'DFTU.ProjectorGenerationMethod': 2,
        '%block DFTU.Proj': [
            'Ni 1',
            'n=3 2',
            '5.3 0.0',
            '0.0 0.0',
        ],
    }
)
```

## Important: What NOT to Do

### ❌ WRONG - Don't wrap in fdf_arguments
```python
# DON'T DO THIS!
user_params={
    'fdf_arguments': {  # ❌ Wrong!
        'PAO.BasisSize': 'TZP',
    }
}
```

### ✅ CORRECT - Pass directly
```python
# DO THIS!
user_params={
    'PAO.BasisSize': 'TZP',  # ✅ Correct!
}
```

### ❌ WRONG - VASP parameters in SIESTA
```python
# DON'T DO THIS! (VASP syntax, not SIESTA)
user_params={
    'LDAU': {...},              # ❌ VASP parameter!
    'LDAU.Projectors': 'Bessel' # ❌ Doesn't exist in SIESTA!
}
```

### ✅ CORRECT - SIESTA DFT+U syntax
```python
# DO THIS! (SIESTA syntax)
user_params={
    'DFTU.ProjectorGenerationMethod': 2,  # ✅ Correct!
    '%block DFTU.Proj': [...],            # ✅ Correct!
}
```

## CLI Tools

Discover presets and recipes:

```bash
# List all material-specific presets
atomate2siesta-presets list

# View preset details
atomate2siesta-presets show surface_metal

# Search by category
atomate2siesta-presets category 2d
atomate2siesta-presets category magnetic

# List all recipes
atomate2siesta-recipe list

# View recipe details
atomate2siesta-recipe show phonon_workflow

# Compare before/after code
atomate2siesta-recipe compare phonon_workflow
```

## Parameter Priority

When combining customization methods:

```
tier defaults  →  preset parameters  →  user_params
(lowest)                                (highest priority)
```

Example:
```python
flow = RecipeBook.phonon_workflow(
    structure,
    tier='advanced',           # Sets base parameters
    preset='phonon_standard',  # Overrides some tier params
    user_params={              # Overrides everything
        'Mesh.Cutoff': '500 Ry'
    }
)
```

Result: `Mesh.Cutoff='500 Ry'` (from user_params, highest priority)

## All Recipes Accept These Parameters

```python
RecipeBook.<recipe_name>(
    structure,                # Required
    user_params={...},        # FDF parameter overrides
    tier='intermediate',      # Computational level
    preset='preset_name',     # Material-specific preset
    dry_run=False,            # Generate inputs only
    # ... recipe-specific parameters
)
```

## Next Steps

1. **Start simple**: Try `01_user_params.py`
2. **Learn presets**: Try `03_presets.py`, explore with `atomate2siesta-presets list`
3. **Combine methods**: Try `04_combining_methods.py` (RECOMMENDED!)
4. **Advanced topics**: Try `05_magnetic_dftu.py` or `06_vdw_2d_materials.py`

## Additional Resources

- **Main guide**: `../RECIPE_CUSTOMIZATION_GUIDE.md` (comprehensive reference)
- **Full tutorial**: `../10_customizing_recipes.py` (all examples in one file)
- **Recipe stats**: `atomate2siesta-recipe stats`
- **Code comparisons**: `atomate2siesta-recipe compare <recipe_name>`

## Tips

1. **Always start with a preset** if available for your material type
2. **Use tier** to quickly adjust accuracy (basic → expert)
3. **Override specific params** with user_params only when needed
4. **Use dry_run=True** to preview input files before running
5. **Check generated siesta.fdf** to verify your customizations

Happy customizing! 🚀
