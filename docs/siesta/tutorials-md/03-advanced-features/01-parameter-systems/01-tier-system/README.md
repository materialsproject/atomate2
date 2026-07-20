# Tutorial: Tier-Based Parameter System

**Category**: 03-advanced-features
**Difficulty**: Beginner to Intermediate
**Time**: ~2 min (dry-run), ~10-20 min (full calculation)
**Prerequisites**: Completed [01-basics](../../01-basics/)

---

## Overview

The tier system provides a 5-level hierarchy for automatic parameter management in atomate2siesta. Instead of manually specifying individual parameters, you can select a tier level that automatically activates appropriate dataclass modules and sets recommended parameters.

### What You'll Learn

- Understanding the 5-tier hierarchy (basic → intermediate → advanced → expert → dirty/ultra)
- How tier levels control module activation
- Using tier presets for material-specific parameters
- Customizing tier-based workflows
- Combining tiers with custom parameters

---

## Tier Levels

### Five Core Tiers

| Tier | Modules | Use Case | Typical Runtime |
|------|---------|----------|-----------------|
| **basic** | 6 core modules | Standard calculations | 1× (baseline) |
| **intermediate** | 8 modules | Production quality | 1.5× |
| **advanced** | 10 modules | High accuracy | 2-3× |
| **expert** | 12 modules | Publication/validation | 3-5× |
| **dirty** | Minimal (maps to basic) | Fast testing | 0.3× |
| **ultra** | All modules (maps to expert) | Maximum accuracy | 5-10× |

### Automatic Module Activation

Each tier level automatically enables appropriate dataclass modules:

- **basic**: Core parameters (basis, k-points, mesh cutoff, SCF, solution method, XC functional)
- **intermediate**: + Mixing parameters, electronic temperature
- **advanced**: + Spin settings, DFT+U, vdW corrections
- **expert**: + Advanced SCF options, optimization algorithms, performance tuning

---

## Tutorial Files

This directory contains 5 focused examples demonstrating tier system usage:

### 1. Basic Tier Usage

**`01_basic_tier.py`**: Basic tier with core modules
- Minimal parameter set for standard calculations
- Comments show how to switch between tier levels
- Good starting point for learning the tier system

### 2. Material-Specific Presets

**`02_preset_surface.py`**: Using tier presets
- Demonstrates material-specific presets
- Example: surface calculations with optimized parameters
- Shows how to apply presets with `apply_tier_preset()`

### 3. Custom Module Control

**`03_custom_modules.py`**: Fine-grained module activation control
- **Comprehensive 65-line docstring explaining module system**
- Manually enable/disable specific modules with `enabled_modules`
- Override default tier behavior
- Lists all 28 available modules organized by tier
- 5 use cases: minimal inputs, testing, hybrid configs, debugging, custom workflows

### 4. Disable Specific Modules

**`04_intermediate_no_spin.py`**: Intermediate tier without spin
- Disable specific modules while keeping tier
- Useful for non-magnetic systems
- Shows `disabled_modules` parameter

### 5. Complete Custom Configuration

**`05_custom_parameters.py`**: Full control over parameters
- Pass `user_params` directly to class methods (not input_set_generator!)
- Demonstrates parameter priority and override logic
- Examples for metallic, large, magnetic, and high-accuracy systems

---

## Quick Start

### 1. Basic Usage

```python
from atomate2.siesta.jobs.core import RelaxMaker

# Use tier level directly
maker = RelaxMaker.fixed_cell_relaxation(tier="basic")
job = maker.make(structure)
```

### 2. Preview with Dry-Run

```python
# Preview what parameters will be used
maker = RelaxMaker.fixed_cell_relaxation(tier="basic", dry_run=True)
job = maker.make(structure)
results = run_locally(job, create_folders=True)
```

### 3. Run Tutorial

```bash
cd tutorials/03-advanced-features/01-tier-system
python 01_basic_tier.py  # Start with basic tier
```

---

## Understanding Module Activation

### How Tiers Work

When you specify a tier level:

1. **Module Selection**: System determines which dataclass modules to activate
2. **Parameter Generation**: Each active module contributes its parameters
3. **Validation**: Parameters are validated against SIESTA FDF schema
4. **FDF Generation**: Final FDF file is created with all parameters

### Module Hierarchy

```
basic (6 modules)
└─ BasisSettings
└─ KpointsSettings
└─ MeshSettings
└─ SCFSettings
└─ SolutionMethodSettings
└─ XCSettings

intermediate (+ 2 modules)
└─ MixingSettings
└─ ElectronicTemperatureSettings

advanced (+ 2 modules)
└─ SpinSettings
└─ DFTUSettings
└─ VdWSettings

expert (+ 2 modules)
└─ AdvancedSCFSettings
└─ OptimizationSettings
└─ PerformanceSettings
```

---

## Parameter Priority

When combining tiers with custom parameters:

1. **Tier baseline**: Tier level sets default parameters
2. **User overrides**: `user_params` override tier defaults
3. **Powerups**: Powerup functions can modify parameters after creation

### Example

```python
# Tier sets default k-points to [4,4,4]
# User overrides to [6,6,6]
maker = RelaxMaker.fixed_cell_relaxation(
    tier="basic",
    user_params={"a2s_kpts": [6, 6, 6]}  # Override tier default
)
```

---

## Common Use Cases

### Case 1: Quick Testing

```python
# Use dirty tier for fast testing
maker = RelaxMaker.fixed_cell_relaxation(
    tier="dirty",  # Minimal parameters, fast runtime
    dry_run=True
)
```

### Case 2: Production Calculations

```python
# Use intermediate tier for production
maker = RelaxMaker.fixed_cell_relaxation(
    tier="intermediate"  # Balanced accuracy/speed
)
```

### Case 3: High-Accuracy Validation

```python
# Use expert tier for validation
maker = RelaxMaker.fixed_cell_relaxation(
    tier="expert"  # Maximum accuracy
)
```

### Case 4: Material-Specific Presets

```python
from atomate2.siesta.sets.tiers import apply_tier_preset

# Use tier preset (recommended)
maker = RelaxMaker.fixed_cell_relaxation()
maker = apply_tier_preset(maker, "relax_standard")
# Preset contains tier + additional recommended parameters
```

---

## Best Practices

1. **Start with basic**: Use `tier="basic"` for initial calculations
2. **Use presets**: Prefer tier presets over manual tier levels (see [06-tier-presets-customization](../06-tier-presets-customization/))
3. **Preview first**: Always use `dry_run=True` to preview parameters
4. **Document tier**: Keep track of which tier you used for reproducibility
5. **Override sparingly**: Only override tier defaults when necessary
6. **Test convergence**: Verify results don't depend on tier level for your system

---

## Tier vs Preset

### Tier Levels (This Tutorial)
- **What**: Control module activation hierarchy
- **When**: Internal parameter system, basic control
- **Example**: `tier="basic"` activates 6 core modules

### Tier Presets ([06-tier-presets-customization](../06-tier-presets-customization/))
- **What**: Material-specific parameter collections
- **When**: Recommended for most users, includes tier + optimized params
- **Example**: `"relax_standard"` preset contains `tier="basic"` + recommended parameters

**Recommendation**: Use tier presets (tutorial 06) instead of raw tier levels for production calculations.

---

## Common Issues

### Issue 1: "Invalid tier level"

**Symptoms**:
```
ValueError: Invalid tier 'basic_dirty'. Must be one of ['basic', 'intermediate',
'advanced', 'expert', 'all', 'dirty', 'ultra']
```

**Solution**: Use tier **level**, not preset **name**:
```python
# ❌ Wrong: preset name
tier="relax_standard"

# ✅ Correct: tier level
tier="basic"

# ✅ Better: Use preset (see tutorial 06)
from atomate2.siesta.sets.tiers import apply_tier_preset
maker = apply_tier_preset(maker, "relax_standard")
```

### Issue 2: "Parameters not applied"

**Symptoms**: Custom parameters don't appear in FDF file

**Cause**: User parameters passed to maker before tier applied

**Solution**: Use `apply_tier_preset()` with `override_params`:
```python
# ❌ Wrong: preset overwrites user_params
maker = RelaxMaker(user_params={"kpts": [6,6,6]})
maker = apply_tier_preset(maker, "relax_standard")  # Overwrites!

# ✅ Correct: use override_params
maker = RelaxMaker()
maker = apply_tier_preset(
    maker,
    "relax_standard",
    override_params={"kpts": [6,6,6]}  # Merges with preset
)
```

### Issue 3: "Too many parameters activated"

**Symptoms**: FDF file has unexpected parameters

**Solution**: Use lower tier level or disable specific modules:
```python
# Use basic tier instead of expert
tier="basic"  # Only 6 core modules

# Or disable specific modules (advanced)
# See tutorial 03_custom_modules.py
```

---

## Related Tutorials

- **[06-tier-presets-customization](../06-tier-presets-customization/)**: Material-specific tier presets (recommended next step)
- **[03-powerups](../03-powerups/)**: Dynamic parameter modification
- **[02-fdf-block-inputs](../02-fdf-block-inputs/)**: Advanced FDF parameters

---

## CLI Tools

```bash
# List all tier presets
atomate2siesta-presets list

# Show preset details
atomate2siesta-presets show relax_standard

# Search by tier level
atomate2siesta-presets search --tier basic

# List presets by category
atomate2siesta-presets category 2d
```

---

## Summary

**What we covered**:
- ✅ 5-tier hierarchy (basic → intermediate → advanced → expert + dirty/ultra)
- ✅ Automatic module activation based on tier
- ✅ Using tier levels in makers
- ✅ Parameter priority and override logic
- ✅ Difference between tiers and presets

**Key takeaways**:
1. Tiers control which dataclass modules are activated
2. Higher tiers activate more modules (more parameters, slower calculations)
3. Start with `tier="basic"` for standard calculations
4. Prefer tier presets over raw tier levels (see tutorial 06)
5. Use `dry_run=True` to preview parameters before running

**Next step**: Proceed to [06-tier-presets-customization](../06-tier-presets-customization/) to learn about material-specific presets.

---

*Back to [03-advanced-features](../README.md) | [Main Tutorial Index](../../README.md)*
