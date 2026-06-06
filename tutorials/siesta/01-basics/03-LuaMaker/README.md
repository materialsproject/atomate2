# Tutorial: Lua Scripts for Advanced Features

**Category**: 01-basics
**Difficulty**: Intermediate
**Time**: ~2 min (dry-run), ~15-20 min (full calculation)

---

## Overview

Learn how to use Lua scripting in SIESTA for advanced geometry optimization and constraints using FLOS (Flexible Lua Objects for Siesta). Lua scripts provide more control than built-in SIESTA relaxation methods.

---

## What You'll Learn

- Using `LuaMaker` for Lua-enabled calculations
- FLOS library for geometry optimization
- Common Lua scripts (LBFGS, FIRE, NEB)
- Advanced geometry constraints via Lua
- Verifying SIESTA has Lua support

---

## Prerequisites

- **Required**: [01-relaxation](../01-RelaxMaker/) completed
- **Required**: SIESTA compiled with Lua support
- **Required**: FLOS library available
- **Recommended**: [02-relaxation-parameters](../02-BandStructureMaker/)

---

## Key Concepts

### FLOS (Flexible Lua Objects for Siesta)

FLOS is a Lua library that provides advanced optimization algorithms for SIESTA:

**Common Lua Scripts**:
- `relax_geometry_lbfgs.lua` - L-BFGS optimizer (memory efficient)
- `relax_geometry_fire.lua` - FIRE algorithm (good for hard cases)
- `neb.lua` - Nudged elastic band calculations
- `constrained_relax.lua` - Custom geometric constraints

**Advantages**:
- ✓ More optimization algorithms than built-in SIESTA
- ✓ Custom constraints (distances, angles, planes)
- ✓ On-the-fly parameter adjustment
- ✓ Advanced NEB calculations
- ✓ Better convergence for difficult systems

### LuaMaker vs RelaxMaker

**RelaxMaker**:
- Uses built-in SIESTA optimization (CG, Broyden, etc.)
- Simpler setup, no external dependencies
- Good for most standard relaxations

**LuaMaker**:
- Uses FLOS Lua scripts for optimization
- Requires SIESTA with Lua support
- Better for advanced cases (NEB, custom constraints, difficult convergence)

---

## Tutorial Files

This directory contains:

1. **`LuaMaker_basic.py`** - Basic Lua-enabled relaxation using FLOS library

---

## Alternative: Using CLI Tool

The CLI tool can also generate Lua-based relaxation scripts:

```bash
# Generate Lua-enabled relaxation script
atomate2siesta-maker relax Si.cif --use-lua

# Or use interactive mode
atomate2siesta-maker --interactive
# (Select "Use Lua scripts" when prompted)
```

**Note**: CLI-generated scripts will check for Lua support and warn if not available.

See the [CLI Tools documentation](../../../docs/source/cli-tools.rst) for more options.

---

## Configuration Options

### Lua Script Selection

```python
LUA_SCRIPT = "relax_geometry_lbfgs.lua"  # Or other FLOS scripts
```

**Available Scripts**:
- **LBFGS**: L-BFGS quasi-Newton (memory efficient, fast)
- **FIRE**: Fast Inertial Relaxation Engine (robust for hard cases)
- **CG**: Conjugate gradient (traditional method)
- **NEB**: Nudged elastic band (transition state searches)

---

## Quick Start

```bash
# 1. Check if SIESTA has Lua support
siesta --version | grep -i lua

# 2. Preview with dry-run
# Edit tutorial.py: RUN_MODE = "dry_run"
python tutorial.py

# 3. Verify Lua.Script parameter in siesta.fdf
cat preview_output/job_*/siesta.fdf | grep "Lua.Script"

# 4. Check Lua script is copied
ls preview_output/job_*/*.lua

# 5. Run calculation (if SIESTA has Lua)
# Edit tutorial.py: RUN_MODE = "local"
python tutorial.py
```

---

## Expected Output

### Dry-Run Mode

```
✅ Dry-run complete!

💡 Check siesta.fdf for Lua.Script parameter:
   grep 'Lua.Script' preview_output/job_*/siesta.fdf

💡 Lua script should be in job folder:
   ls preview_output/job_*/*.lua
```

**Files Generated**:
```
preview_output/job_*/
├── siesta.fdf                    # Contains: Lua.Script relax_geometry_lbfgs.lua
├── structure.fdf
├── relax_geometry_lbfgs.lua      # Lua optimization script
├── *.psml
└── structure.cif
```

### Local Mode

```
✅ Calculation complete!

📊 Results:
  - Job folder contains optimized structure
  - Lua optimization algorithm was used
  - Check siesta.out for Lua messages
```

**Look for in siesta.out**:
```
Lua: Running relaxation with LBFGS
Lua: Step    1, Force = 0.123 eV/Ang
Lua: Step    2, Force = 0.089 eV/Ang
...
Lua: Converged in 15 steps
```

---

## Common Issues

### Issue 1: "Lua not supported in this SIESTA build"

**Symptoms**:
```
siesta: ERROR: Lua scripting requested but not compiled in
```

**Solution**: Install SIESTA with Lua support
```bash
# Using conda
conda install -c conda-forge siesta

# Or recompile SIESTA with --enable-lua flag
```

**Check support**:
```bash
siesta --version | grep -i lua
# Should show: "Lua support: yes"
```

### Issue 2: "FLOS library not found"

**Symptoms**:
```
Lua: ERROR: module 'flos' not found
```

**Solution**: Install FLOS library
```bash
# Clone FLOS repository
git clone https://github.com/siesta-project/flos.git

# Set FLOS_PATH in ~/.atomate2siesta.yaml
FLOS_PATH: "/path/to/flos"
```

### Issue 3: "Lua script not found"

**Symptoms**:
```
Lua: ERROR: cannot open relax_geometry_lbfgs.lua
```

**Solution**: Ensure script is in calculation directory
- Dry-run mode automatically copies script
- For manual runs, copy from FLOS examples:
  ```bash
  cp /path/to/flos/examples/relax_geometry_lbfgs.lua .
  ```

### Issue 4: "Optimization not converging"

**Solution**: Try different Lua scripts
1. **FIRE algorithm**: Better for difficult cases
   ```python
   LUA_SCRIPT = "relax_geometry_fire.lua"
   ```

2. **Adjust force tolerance** in Lua script:
   ```lua
   -- Edit .lua file
   tol_force = 0.04  -- eV/Ang (default)
   ```

3. **Increase max steps** in siesta.fdf:
   ```python
   "MD.NumCGsteps": 200  # default: 100
   ```

### Issue 5: "Unknown FDF parameter: fdf_arguments"

**Error**: `ValueError: Unknown FDF parameter(s): fdf_arguments`

**Cause**: Using deprecated `fdf_arguments` wrapper syntax

**Solution**:
```python
# ❌ OLD (doesn't work):
user_params = {
    "fdf_arguments": {
        "DM.InitSpin": [...]
    }
}

# ✅ NEW (correct):
user_params = {
    "%block DM.InitSpin": [...]  # Directly in user_params!
}
```

**Note**: Block parameters should be specified **directly** in `user_params`, NOT nested in `fdf_arguments`. See [FDF Block Parameters](#fdf-block-parameters-advanced) section above.

---

## Verifying Lua Setup

### Check SIESTA Lua Support

```bash
# Method 1: Check version info
siesta --version | grep -i lua

# Method 2: Run simple Lua test
cat > test.fdf <<EOF
SystemName Test
SystemLabel test
Lua.Script test.lua
EOF

cat > test.lua <<EOF
print("Lua works!")
EOF

siesta < test.fdf
# Should print "Lua works!" in output
```

### Check FLOS Installation

```bash
# Verify FLOS_PATH is set
grep FLOS_PATH ~/.atomate2siesta.yaml

# Check Lua can find FLOS
lua -e "package.path='/path/to/flos/?.lua;' .. package.path; require 'flos'"
# Should not error
```

---

## Lua Script Customization

### Basic Structure

```lua
-- relax_geometry_lbfgs.lua structure
local flos = require "flos"

-- Create LBFGS optimizer
local optimizer = flos.LBFGS{H0 = 1.0 / 75.0}

function siesta_comm()
  -- Called by SIESTA at each MD step
  -- Read forces, update positions
  -- Return new coordinates
end
```

### Custom Constraints

```lua
-- Fix atom z-coordinate
function siesta_comm()
  -- ... get forces and positions ...

  -- Fix z-coordinate of atom 1
  F[1][3] = 0.0
  R[1][3] = initial_z

  -- Continue optimization
end
```

---

## Comparison: Built-in vs Lua

### When to Use Built-in (RelaxMaker)

✓ Simple relaxations
✓ Standard systems
✓ No special constraints
✓ Faster setup (no Lua dependency)

### When to Use Lua (LuaMaker)

✓ Difficult convergence cases
✓ Custom geometric constraints
✓ NEB calculations
✓ Advanced optimization algorithms
✓ Research requiring specific methods

---

## Advanced Customization

For more sophisticated parameter control and workflow customization:

**📖 Parameter Customization Methods**
- [Makers vs FlowMakers](../../../docs/source/makers-vs-flowmakers.rst) - Comprehensive guide on when to use `user_params`, tier presets, or powerups
- [Powerups System](../../../docs/source/features.rst#powerups-system) - Runtime parameter modifications for jobs and flows
- [Tier System](../../../docs/source/tier-system.rst) - Material-specific parameter presets

**🎯 Quick Examples**

*Using tier presets with Lua:*
```python
from atomate2.siesta.sets.tiers import apply_tier_preset

maker = LuaMaker.fixed_cell_relaxation(lua_script="relax_geometry_lbfgs.lua")
maker = apply_tier_preset(maker, "relax_standard")
```

*Using powerups for runtime modifications:*
```python
from atomate2.siesta.powerups import update_user_siesta_settings

maker = LuaMaker.fixed_cell_relaxation(dry_run=True)
job = maker.make(structure)
job = update_user_siesta_settings(job, {
    "MD.MaxForceTol": "0.02 eV/Ang",
    "MD.NumCGsteps": 200,
})
```

*Combining Lua with custom parameters:*
```python
maker = LuaMaker.fixed_cell_relaxation(
    lua_script="relax_geometry_fire.lua",
    user_params={
        "PAO.BasisSize": "DZP",
        "Mesh.Cutoff": "300 Ry",
    }
)
```

---

## Next Steps

After completing this tutorial:

1. **Try different Lua scripts**: Test FIRE vs LBFGS performance
2. **Multi-step workflows**: [05-workflows](../04-RelaxMaker-StaticMaker/)
3. **NEB calculations**: [03-advanced-workflows/04-neb](../../../02-workflows/04-neb/) - Nudged elastic band with Lua
4. **Custom constraints**: Modify Lua scripts for your needs
5. **Advanced features**: Explore FLOS examples for complex optimizations

---

## FDF Block Parameters (Advanced)

When using LuaMaker with custom FDF block parameters, use the `"%block ParamName"` syntax **directly** in `user_params`.

**IMPORTANT**: DO NOT wrap block parameters in `fdf_arguments` - this is deprecated!

### Correct Usage

```python
# ✅ CORRECT: Block parameters directly in user_params
from atomate2.siesta.jobs.core import LuaMaker

maker = LuaMaker.fixed_cell_relaxation(
    lua_script="relax_geometry_lbfgs.lua",
    user_params={
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [6, 6, 6],

        # Geometry constraints (if needed with Lua)
        "%block Geometry.Constraints": [
            "position from 1 to 3",
        ],
    }
)
```

### Incorrect Usage (Deprecated)

```python
# ❌ WRONG: Don't nest in fdf_arguments!
maker = LuaMaker.fixed_cell_relaxation(
    user_params={
        "fdf_arguments": {  # <-- This doesn't work!
            "Geometry.Constraints": [...]
        }
    }
)
```

**Common block parameters**:
- `"%block DM.InitSpin"` - Initial magnetic moments
- `"%block Geometry.Constraints"` - Fix atoms (complementary to Lua constraints)
- `"%block DFTU.Proj"` - DFT+U projectors

For comprehensive examples, see [02-fdf-block-inputs](../../03-advanced-features/01-parameter-systems/04-fdf-blocks/).

---

## Tips for Success

✅ **Verify Lua support first**: Don't waste time if SIESTA lacks Lua - use dry-run to check
✅ **Start with LBFGS**: Generally most efficient algorithm for standard cases
✅ **Compare with built-in**: Use RelaxMaker as baseline to verify improvements
✅ **Try FIRE for hard cases**: If LBFGS struggles, FIRE is more robust
✅ **Read FLOS docs**: Many advanced features available (constraints, cell optimization)
✅ **Check siesta.out**: Lua prints useful convergence info at each step
✅ **Use dry-run mode**: Verify Lua script is copied correctly before running
✅ **Block parameters**: Use `"%block ParamName"` directly in `user_params` - NO `fdf_arguments` wrapper!

---

## Additional Resources

**FLOS Documentation**:
- GitHub: https://github.com/siesta-project/flos
- Examples: Check `flos/examples/` directory
- API reference: See FLOS documentation

**SIESTA Manual**:
- Chapter 10: Lua scripting
- Section 4.7: Geometry optimization

**Lua Resources**:
- Lua 5.3 manual: https://www.lua.org/manual/5.3/
- Lua quick reference

---

*Back to [01-basics](../README.md) | [Main Tutorial Index](../../README.md)*
