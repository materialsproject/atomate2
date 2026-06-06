# Tutorial: Structural Relaxation with RelaxMaker

**Category**: 01-basics/01-RelaxMaker
**Difficulty**: Beginner
**Time**: ~2 min (dry-run), ~10-15 min (full calculation)

---

## Overview

This tutorial introduces structural relaxation using `RelaxMaker`, the fundamental building block for atomate2siesta calculations. Structural relaxation optimizes atomic positions (and optionally cell parameters) to minimize forces and stress, finding the lowest-energy configuration.

**Why Relaxation is Essential**:
- Initial structures often have non-equilibrium geometries
- Bond lengths and angles need optimization
- Cell parameters may need adjustment (variable-cell relaxation)
- Required before accurate property calculations (band structure, DOS, phonons)

---

## What You'll Learn

- Using `RelaxMaker.fixed_cell_relaxation()` - relax atomic positions only
- Using `RelaxMaker.variable_cell_relaxation()` - relax atoms + cell parameters
- Parameter customization with `user_params`
- FDF block parameter syntax (`"%block DM.InitSpin"`, etc.)
- Magnetic moment specification (automatic and explicit)
- Automatic XC functional detection from pseudopotentials
- Remote job submission with jobflow-remote

---

## Prerequisites

- **SIESTA installed and configured**: `~/.atomate2siesta.yaml` set up
- **Pseudopotentials installed**: Use `atomate2siesta-pseudos install ONCVPSP-PBE-SR-PDv0.4-Standard`
- **Structure files**: Located in [00-structures](../../00-structures/)
- **Python 3.9+** with atomate2siesta installed

---

## Key Concepts

### Fixed-Cell vs Variable-Cell Relaxation

**Fixed-Cell Relaxation**:
- Optimizes atomic positions only
- Cell shape and volume remain constant
- Faster, suitable when cell is already optimized
- Use for: Surface slabs, molecules in boxes, pre-optimized cells

**Variable-Cell Relaxation**:
- Optimizes both atomic positions AND cell parameters
- Cell shape and volume can change
- Slower, required for unknown systems
- Use for: Bulk materials, finding equilibrium lattice parameters

### Convergence Criteria

Relaxation stops when forces fall below threshold:
- **Default**: 0.04 eV/Å (reasonable for most purposes)
- **Tight**: 0.01 eV/Å (high-accuracy calculations)
- **Loose**: 0.1 eV/Å (quick tests only)

Controlled by: `MD.MaxForceTol` parameter

---

## Tutorial Files

This directory contains 11 progressive examples:

### Basic Usage
1. **`RelaxMaker_fixed_cell.py`** - Simple fixed-cell relaxation
2. **`RelaxMaker_variable_cell.py`** - Variable-cell relaxation
3. **`RelaxMaker_fixed_cell_oneliner.py`** - One-line fixed-cell workflow
4. **`RelaxMaker_variable_cell_oneliner.py`** - One-line variable-cell workflow

### Parameter Customization
5. **`RelaxMaker_custom_params.py`** - Custom SIESTA parameters (basis, k-points, cutoff)
6. **`RelaxMaker_custom_params_magnetic.py`** - Spin-polarized relaxation with automatic magmom detection

### Advanced Features
7. **`RelaxMaker_auto_xc_detection_1.py`** - Automatic XC functional detection (basic)
8. **`RelaxMaker_auto_xc_detection_2.py`** - XC detection with validation
9. **`RelaxMaker_auto_xc_detection_3.py`** - XC detection with error handling

### Magnetic Systems
10. **`RelaxMaker_explicit_dminitspin.py`** - Manual DM.InitSpin specification with `"%block DM.InitSpin"`

### Remote Execution
11. **`RelaxMaker_custom_params_jobflow_remote.py`** - Submit to HPC cluster

---

## Quick Start

### Example 1: Basic Fixed-Cell Relaxation

```python
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

# Load structure
structure = Structure.from_file("../../00-structures/Si.cif")

# Create maker and run
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)  # Preview mode
job = maker.make(structure)
results = run_locally(job, create_folders=True)
```

**What happens**:
1. Creates job folder with SIESTA input files
2. In dry-run mode: No calculation, just previews setup
3. With `dry_run=False`: Runs SIESTA relaxation (~10 min for Si)

### Example 2: Variable-Cell Relaxation

```python
# Optimize both atoms AND cell
maker = RelaxMaker.variable_cell_relaxation(dry_run=True)
job = maker.make(structure)
results = run_locally(job, create_folders=True)
```

### Example 3: Custom Parameters

```python
maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "PAO.BasisSize": "DZP",       # Double-zeta + polarization
        "kpts": [6, 6, 6],            # K-point mesh
        "Mesh.Cutoff": "300 Ry",      # Real-space grid cutoff
        "MD.MaxForceTol": "0.02 eV/Ang",  # Tight convergence
    },
    dry_run=True
)
```

### Example 4: Magnetic Relaxation (Automatic)

```python
from atomate2.siesta.sets.utils import get_default_initial_magnetic_moments

# Auto-detect magnetic moments (Fe, Co, Ni, etc.)
magmoms = get_default_initial_magnetic_moments(structure)
structure.add_site_property("magmom", magmoms)

maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "Spin": "polarized",
        "a2s_magnetic_ordering": "ferromagnetic",  # or "AFM"
    },
    dry_run=True
)
job = maker.make(structure)
```

### Example 5: Magnetic Relaxation (Explicit DM.InitSpin)

```python
# Manual control over initial spin configuration
maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [4, 4, 4],
        "Spin": "polarized",
        # Explicitly define DM.InitSpin block (no fdf_arguments wrapper!)
        "%block DM.InitSpin": [
            "1  +2.0",  # First atom: +2.0 μB
            "2  -2.0",  # Second atom: -2.0 μB (antiferromagnetic)
        ],
    },
    dry_run=True
)
```

**Note**: Use `"%block DM.InitSpin"` directly in `user_params` - NO `fdf_arguments` wrapper needed!

---

## Run Modes

### 1. Dry-Run Mode (Always Start Here!)

```bash
# Edit tutorial script: Set dry_run=True
python 01_fixed_cell.py
```

**Output**:
```
preview_output/job_*/
├── siesta.fdf           # SIESTA input parameters
├── structure.fdf        # Atomic structure
├── *.psml               # Pseudopotential files
└── structure.cif        # Structure in CIF format
```

**Use for**:
- Verifying parameter settings before running
- Checking pseudopotential files are found
- Inspecting generated FDF files

### 2. Local Execution

```bash
# Edit: Set dry_run=False
python 01_fixed_cell.py
```

**Output**:
```
job_*/
├── siesta.fdf
├── siesta.out           # SIESTA output (check for convergence)
├── siesta.XV            # Final optimized structure
├── siesta.DM            # Density matrix (for restart)
└── [many more SIESTA files]
```

**Time**: ~5-15 minutes for small systems (Si, MgO)

### 3. HPC Cluster Submission

See `09_custom_params_jobflow_remote.py` for jobflow-remote setup.

---

## Expected Output

### Successful Relaxation

```bash
grep "Relaxed" job_*/siesta.out
# Output: "Relaxed in 15 steps"
```

**Check convergence**:
```bash
# Final forces should be < 0.04 eV/Å
tail -50 job_*/siesta.out | grep "siesta: Atomic forces"
```

**Extract final structure**:
```python
from pymatgen.io.siesta import SiestaOutput

output = SiestaOutput("job_*/siesta.out")
relaxed_structure = output.final_structure
print(f"Final volume: {relaxed_structure.volume:.3f} Å³")
```

### Comparing Initial vs Final Structure

```python
from pymatgen.core import Structure

initial = Structure.from_file("../../00-structures/Si.cif")
final = Structure.from_file("job_*/siesta.XV")

print(f"Volume change: {final.volume - initial.volume:.3f} Å³")
print(f"Volume change %: {(final.volume / initial.volume - 1) * 100:.2f}%")
```

---

## Common Issues

### Issue 1: "Relaxation not converging"

**Symptoms**: Job runs for 100+ steps without converging

**Solutions**:
1. **Increase max steps**:
   ```python
   user_params={"MD.NumCGsteps": 200}  # Default: 100
   ```

2. **Relax convergence criteria**:
   ```python
   user_params={"MD.MaxForceTol": "0.1 eV/Ang"}  # Looser for testing
   ```

3. **Check for instabilities**:
   ```bash
   grep "SCF Convergence" job_*/siesta.out
   # If SCF not converging, may need better k-points or mixer settings
   ```

4. **Try different optimizer**:
   - Use `LuaMaker` with FIRE algorithm (see [03-LuaMaker](../03-LuaMaker/))

### Issue 2: "Forces still high after 'convergence'"

**Cause**: Unconverged parameters (k-points, basis, cutoff)

**Solution**: Run convergence tests first (see [02-workflows/01-convergence](../../02-workflows/01-convergence/))

### Issue 3: "Cell explodes" (variable-cell only)

**Symptoms**: Lattice parameters become unrealistically large or small

**Solutions**:
1. **Check initial structure quality**:
   ```python
   # Atoms too close?
   structure.get_all_neighbors(r=1.0)  # Should return empty if no close contacts
   ```

2. **Use smaller stress tolerance**:
   ```python
   user_params={"MD.MaxStressTol": "0.5 GPa"}  # Default: 1 GPa
   ```

3. **Fix problematic cell parameters**:
   ```python
   # If only a-axis is problematic, use selective relaxation
   # See 02-fdf-block-inputs for geometry constraints
   ```

### Issue 4: "Pseudopotential not found"

**Error**: `ERROR: Could not find pseudopotential for Si`

**Solution**:
```bash
# Install pseudopotentials
atomate2siesta-pseudos install ONCVPSP-PBE-SR-PDv0.4-Standard

# Configure path in ~/.atomate2siesta.yaml
echo "SIESTA_PP_PATH: ~/.atomate2siesta/pseudos/ONCVPSP-PBE-SR-PDv0.4-Standard" >> ~/.atomate2siesta.yaml
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

**Note**: Block parameters like `%block DM.InitSpin`, `%block Geometry.Constraints`, and `%block DFTU.Proj` should be specified **directly** in `user_params` or `override_params`, NOT nested in `fdf_arguments`.

---

## Alternative: Using CLI Tool

Generate relaxation scripts automatically:

```bash
# Interactive mode (recommended for beginners)
atomate2siesta-maker --interactive
# Select: "relax" → "Fixed cell" → Choose structure → Done!

# Command-line mode
atomate2siesta-maker relax Si.cif
atomate2siesta-maker relax Si.cif --preset relax_standard  # With tier preset
```

**Benefits**:
- No Python coding required
- Step-by-step guided setup
- Automatic preset application
- Choice of local/remote/dry-run execution

See [CLI Tools documentation](../../../docs/source/cli-tools.rst) for all options.

---

## Parameter Customization

### Using Tier Presets (Recommended)

```python
from atomate2.siesta.sets.tiers import apply_tier_preset

maker = RelaxMaker.fixed_cell_relaxation()
maker = apply_tier_preset(maker, "relax_standard")
# Applies material-specific optimized parameters
```

**Available presets** (see [06-tier-presets-customization](../../03-advanced-features/06-tier-presets-customization/)):
- `relax_standard`: Balanced accuracy/speed
- `tight_relax`: High accuracy
- `quick_relax`: Fast testing
- `surface_relax`: For surface slabs
- `2d_relax`: For 2D materials

### Common Parameters to Customize

```python
user_params = {
    # Basis set
    "PAO.BasisSize": "DZP",        # SZ, DZ, SZP, DZP, TZP

    # K-points
    "a2s_kpts": [6, 6, 6],         # Denser = more accurate, slower

    # Real-space grid
    "Mesh.Cutoff": "300 Ry",       # Higher = more accurate, slower

    # Convergence
    "MD.MaxForceTol": "0.02 eV/Ang",  # Tighter = more accurate
    "MD.NumCGsteps": 200,          # Max relaxation steps

    # SCF convergence
    "DM.Tolerance": "1e-4",        # Electronic convergence threshold

    # Spin polarization
    "Spin": "polarized",           # For magnetic systems
}
```

### FDF Block Parameters (Advanced)

**IMPORTANT**: Block parameters use `"%block ParamName"` syntax directly in `user_params`.
**DO NOT** wrap them in `fdf_arguments` - this is deprecated!

```python
# ✅ CORRECT: Block parameters directly in user_params
user_params = {
    "Spin": "polarized",

    # DM.InitSpin block for magnetic moments
    "%block DM.InitSpin": [
        "1  +2.0",
        "2  -2.0",
    ],

    # Geometry constraints (fix atoms)
    "%block Geometry.Constraints": [
        "position from 1 to 3",  # Fix atoms 1-3
    ],

    # DFT+U projectors
    "%block DFTU.Proj": [
        "Cu 1",
        "n=3 2",
        "7.0 0.0",
        "0.0 0.0",
    ],
}

# ❌ WRONG: Don't nest in fdf_arguments!
user_params = {
    "fdf_arguments": {  # <-- This doesn't work!
        "DM.InitSpin": [...]
    }
}
```

**Common FDF Blocks**:
- `"%block DM.InitSpin"` - Initial magnetic moments
- `"%block Geometry.Constraints"` - Fix atoms during relaxation
- `"%block DFTU.Proj"` - DFT+U projectors
- `"%block PAO.Basis"` - Custom orbital definitions

See [02-fdf-block-inputs](../../03-advanced-features/02-fdf-block-inputs/) for comprehensive examples.

---

## Best Practices

✅ **Always start with dry-run**: Catch errors before wasting compute time
✅ **Check convergence**: Verify forces < threshold in siesta.out
✅ **Use tier presets**: Start with `relax_standard`, adjust if needed
✅ **Converge parameters first**: K-points, basis, cutoff (see [02-workflows/01-convergence](../../02-workflows/01-convergence/))
✅ **Save relaxed structures**: Use for subsequent calculations (bands, DOS, phonons)
✅ **Compare initial vs final**: Sanity check that relaxation makes sense
✅ **Block parameters**: Use `"%block ParamName"` directly in `user_params` - NO `fdf_arguments` wrapper!

---

## Next Steps

After completing relaxation tutorials:

1. **Multi-step workflows**: [04-RelaxMaker-StaticMaker](../04-RelaxMaker-StaticMaker/)
2. **Electronic structure**: [02-BandStructureMaker](../02-BandStructureMaker/), [05-DOSMaker](../05-DOSMaker/)
3. **Convergence testing**: [02-workflows/01-convergence](../../02-workflows/01-convergence/)
4. **Magnetic systems**: [03-advanced-features/16-magnetic-calculations](../../03-advanced-features/16-magnetic-calculations/)
5. **One-line workflows**: [03-advanced-features/08-recipe-book](../../03-advanced-features/08-recipe-book/)

---

## Tips for Success

✅ **Preview first**: Use dry_run=True to check setup
✅ **Check forces**: Verify convergence in siesta.out
✅ **Start simple**: Use default parameters, customize only if needed
✅ **Document changes**: Keep track of what parameters you modify
✅ **Save outputs**: Relaxed structures needed for property calculations

---

## References

- **SIESTA Manual**: Geometry optimization section
- **Pymatgen**: Structure manipulation and analysis
- **Jobflow**: Workflow management
- **atomate2siesta docs**: [Relaxation guide](../../../docs/source/features.rst)

---

*Back to [01-basics](../README.md) | [Main Tutorial Index](../../README.md)*
