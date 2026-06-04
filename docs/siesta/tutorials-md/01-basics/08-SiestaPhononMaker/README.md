# Tutorial: Native SIESTA Phonon Calculations with SiestaPhononMaker

**Category**: 01-basics/08-SiestaPhononMaker
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~30-60 min (full calculation)

---

## Overview

This tutorial demonstrates phonon calculations using SIESTA's native force constants approach via `SiestaPhononMaker`. Unlike `PhonopyMaker` which uses finite displacements with Phonopy, this method uses SIESTA's built-in capability to compute force constants directly.

**When to Use Native SIESTA Phonons**:
- Quick phonon calculations for small systems
- SIESTA-specific phonon features
- No external Phonopy dependency
- Comparison with Phonopy results

**When to Use PhonopyMaker Instead** (Recommended):
- Complex phonon workflows (QHA, Grüneisen)
- Automatic plotting
- Better post-processing tools
- Industry-standard Phonopy ecosystem

---

## What You'll Learn

- Using `SiestaPhononMaker` for native SIESTA phonons
- SIESTA's force constant calculation method
- MD.TypeOfRun = "phonon" mode
- Analyzing SIESTA phonon output files
- Comparison with PhonopyMaker approach

---

## Prerequisites

- **Required tutorials**: [01-RelaxMaker](../01-RelaxMaker/), [07-PhonopyMaker](../07-PhonopyMaker/)
- **Required knowledge**: Phonon basics, force constants
- **SIESTA configuration**: `~/.atomate2.yaml` set up
- **Comparison recommended**: Complete [07-PhonopyMaker](../07-PhonopyMaker/) first

---

## Key Concepts

### Native SIESTA Phonon Method

SIESTA computes force constants using:

1. **Self-consistent calculation**: Get converged electronic structure
2. **Force constant calculation**: SIESTA perturbs each atom internally
3. **Output**: Harmonic force constants → `.FC` file

**Key FDF Parameters**:
```python
user_params = {
    "MD.TypeOfRun": "phonon",      # Enable phonon mode
    "MD.FCDispl": "0.01 Ang",      # Displacement magnitude
    "MD.FCfirst": 1,               # First atom to displace
    "MD.FClast": None,             # Last atom (None = all)
}
```

### Comparison: Native SIESTA vs Phonopy

| Aspect | Native SIESTA | Phonopy (PhonopyMaker) |
|--------|---------------|------------------------|
| **Method** | Internal force constants | Finite displacements |
| **Calculations** | ~10-20 (fewer) | ~20-50 (more) |
| **Post-processing** | Manual | Automatic (3 plots) |
| **QHA/Grüneisen** | Manual integration | Built-in workflows |
| **Supercell** | Generated internally | Explicit control |
| **Output format** | SIESTA .FC files | Phonopy YAML/HDF5 |
| **Ecosystem** | SIESTA-specific | Standard (works with ASE, etc.) |

**Recommendation**: Use `PhonopyMaker` for production workflows, `SiestaPhononMaker` for quick checks or SIESTA-specific features.

---

## Tutorial Files

This directory contains:

1. **`SiestaPhononMaker_basic.py`** - Basic native SIESTA phonon calculation

---

## Quick Start

### Example 1: Basic Native Phonon Calculation

```python
from pymatgen.core import Structure
from atomate2.siesta.jobs.phonon import SiestaPhononMaker
from jobflow import run_locally

# Load structure (MUST be relaxed!)
structure = Structure.from_file("../../00-structures/Si.cif")

# Create maker
maker = SiestaPhononMaker(
    user_params={
        "PAO.BasisSize": "DZP",
        "kpts": [6, 6, 6],
        "Mesh.Cutoff": "300 Ry",
        "MD.TypeOfRun": "phonon",
        "MD.FCDispl": "0.01 Ang",
    },
    dry_run=True
)

# Generate and run
job = maker.make(structure)
results = run_locally(job, create_folders=True)
```

### Example 2: With Tight Convergence

```python
maker = SiestaPhononMaker(
    user_params={
        "PAO.BasisSize": "DZP",
        "kpts": [8, 8, 8],
        "Mesh.Cutoff": "400 Ry",
        "DM.Tolerance": "1e-6",        # Tight electronic convergence
        "MD.TypeOfRun": "phonon",
        "MD.FCDispl": "0.01 Ang",
        "MD.FCfirst": 1,               # Start from atom 1
        "MD.FClast": None,             # All atoms (None = auto)
    },
    dry_run=True
)
```

### Example 3: After Relaxation

```python
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import Flow

# Relax → Phonon workflow
relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={"MD.MaxForceTol": "0.01 eV/Ang"}
)

phonon_maker = SiestaPhononMaker(
    user_params={
        "MD.TypeOfRun": "phonon",
        "MD.FCDispl": "0.01 Ang",
    }
)

relax_job = relax_maker.make(structure)
phonon_job = phonon_maker.make(structure, prev_dir=relax_job.output.dir_name)

workflow = Flow([relax_job, phonon_job])
results = run_locally(workflow, create_folders=True)
```

---

## Run Modes

### 1. Dry-Run Mode

```bash
python 02_basic.py  # With dry_run=True
```

**Check parameters**:
```bash
grep "MD.TypeOfRun" preview_output/job_*/siesta.fdf
grep "MD.FCDispl" preview_output/job_*/siesta.fdf
```

### 2. Local Execution

```bash
# Edit: Set dry_run=False
python 02_basic.py
```

**Output**:
```
job_*/
├── siesta.fdf
├── siesta.out
├── SystemLabel.FC        # Force constants file
├── SystemLabel.bands     # Phonon band structure (if requested)
└── [SIESTA output files]
```

**Time**: 30-60 minutes for small systems (faster than Phonopy)

---

## Expected Output

### Force Constants File (.FC)

SIESTA outputs force constants in `.FC` format:

```
# Force constant matrix
# Atom_i  Atom_j  x_i  y_i  z_i  x_j  y_j  z_j  FC_xx  FC_xy  ...
    1       1     0.0  0.0  0.0  0.0  0.0  0.0  12.34  0.00  ...
    1       2     0.0  0.0  0.0  1.43 1.43 1.43  -2.45 0.12  ...
```

### Analyzing Force Constants

```python
import numpy as np

# Read force constants
def read_siesta_fc(filename):
    """Read SIESTA .FC file."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            atom_i, atom_j = int(parts[0]), int(parts[1])
            r_i = np.array([float(x) for x in parts[2:5]])
            r_j = np.array([float(x) for x in parts[5:8]])
            fc_matrix = np.array([float(x) for x in parts[8:]]).reshape(3, 3)
            data.append((atom_i, atom_j, r_i, r_j, fc_matrix))
    return data

fc_data = read_siesta_fc("job_*/SystemLabel.FC")

# Check force constant sum rule
total_fc = np.zeros((3, 3))
for atom_i, atom_j, r_i, r_j, fc_matrix in fc_data:
    if atom_i == 1:  # Sum over all j for atom 1
        total_fc += fc_matrix

print("Force constant sum (should be ~0):")
print(total_fc)
print(f"Sum rule violation: {np.abs(total_fc).max():.6f} eV/Ang^2")
```

### Computing Phonon Dispersion

SIESTA can compute phonon bands if requested:

```python
# Add to user_params
user_params = {
    "MD.TypeOfRun": "phonon",
    "BandLinesScale": "pi/a",
    "%block BandLines": [
        "1  0.0 0.0 0.0  L  # Gamma",
        "20 0.5 0.0 0.5  X",
        "20 0.5 0.25 0.75 W",
        "1  0.5 0.5 0.5  # L"
        ]
    }
}
```

Then analyze:
```bash
# SIESTA outputs SystemLabel.bands
grep "Phonon" job_*/siesta.out
```

---

## Common Issues

### Issue 1: "Force constants don't satisfy sum rule"

**Symptoms**: Large residual in sum rule check

**Causes & Solutions**:

1. **Electronic structure not converged**:
   ```python
   user_params = {
       "DM.Tolerance": "1e-6",    # Tighter
       "DM.NumberPulay": 8,
   }
   ```

2. **K-points insufficient**:
   ```python
   user_params = {"a2s_kpts": [10, 10, 10]}  # Denser
   ```

3. **Structure not relaxed**:
   - Always relax with tight tolerance first
   - Check forces < 0.01 eV/Å

### Issue 2: ".FC file not generated"

**Cause**: MD.TypeOfRun not set correctly

**Solution**: Verify in siesta.fdf
```bash
grep "MD.TypeOfRun" job_*/siesta.fdf
# Should show: MD.TypeOfRun phonon
```

### Issue 3: "Phonon frequencies imaginary"

**Same as PhonopyMaker** - see [07-PhonopyMaker](../07-PhonopyMaker/) for solutions:
- Structure not fully relaxed
- Force calculation not converged
- Need tighter parameters

### Issue 4: "How to plot phonon bands?"

**Challenge**: Native SIESTA output requires manual plotting

**Solutions**:

1. **Convert to Phonopy format**:
   - Use utility scripts to convert .FC → FORCE_CONSTANTS
   - Then use Phonopy for plotting

2. **Use PhonopyMaker instead** (Recommended):
   - Automatic plotting
   - Better post-processing
   - See [07-PhonopyMaker](../07-PhonopyMaker/)

3. **Manual plotting**:
   ```python
   # Parse SystemLabel.bands and plot manually
   import matplotlib.pyplot as plt
   # ... custom parsing code ...
   ```

### Issue 5: "Unknown FDF parameter: fdf_arguments"

**Error**: `ValueError: Unknown FDF parameter(s): fdf_arguments`

**Cause**: Using deprecated `fdf_arguments` wrapper syntax

**Solution**:
```python
# ❌ OLD (doesn't work):
user_params = {
    "fdf_arguments": {
        "BandLines": [...]
    }
}

# ✅ NEW (correct):
user_params = {
    "%block BandLines": [...]  # Directly in user_params!
}
```

**Note**: Block parameters should be specified **directly** in `user_params`, NOT nested in `fdf_arguments`. See [FDF Block Parameters](#fdf-block-parameters-advanced) section below.

---

## Comparison Example

### Run Both Methods

```python
from atomate2.siesta.jobs.phonon import PhonopyMaker, SiestaPhononMaker
from jobflow import Flow

structure = Structure.from_file("Si.cif")

# Method 1: Native SIESTA
siesta_phonon = SiestaPhononMaker(
    user_params={"MD.TypeOfRun": "phonon"}
)
job1 = siesta_phonon.make(structure)

# Method 2: Phonopy
phonopy_phonon = PhonopyMaker(min_length=15.0)
job2 = phonopy_phonon.make(structure)

# Run both
workflow = Flow([job1, job2])
results = run_locally(workflow, create_folders=True)
```

**Compare**:
- Calculation time
- Force constant accuracy
- Phonon frequencies
- Ease of analysis

**Expected**: Frequencies should agree within ~5%, but Phonopy provides better post-processing.

---

## When to Use Native SIESTA Phonons

### Good Use Cases ✓

- **Quick checks**: Fast phonon calculation for stability
- **Small systems**: < 10 atoms, simple structure
- **SIESTA-specific features**: Using SIESTA's unique phonon capabilities
- **No Phonopy**: When Phonopy not available

### Better with PhonopyMaker ✓✓✓

- **Production workflows**: Publication-quality results
- **Automatic plotting**: 3 plots generated automatically
- **QHA/Grüneisen**: Built-in workflows
- **Large systems**: Better scaling and post-processing
- **Complex analysis**: Phonopy ecosystem integration

---

## Converting Native SIESTA Output

### To Phonopy Format

If you need Phonopy features after native SIESTA calculation:

```python
def convert_siesta_fc_to_phonopy(fc_file, structure):
    """Convert SIESTA .FC to Phonopy FORCE_CONSTANTS."""
    # Read SIESTA force constants
    fc_data = read_siesta_fc(fc_file)

    # Build Phonopy force constant matrix
    n_atoms = len(structure)
    fc_matrix = np.zeros((n_atoms, n_atoms, 3, 3))

    for atom_i, atom_j, r_i, r_j, fc in fc_data:
        fc_matrix[atom_i-1, atom_j-1, :, :] = fc

    # Write Phonopy format
    np.save("FORCE_CONSTANTS.npy", fc_matrix)

    # Can now use Phonopy for analysis
    from phonopy import Phonopy
    from phonopy.interface.calculator import get_default_physical_units

    phonon = Phonopy(structure, [[1,0,0],[0,1,0],[0,0,1]])
    phonon.set_force_constants(fc_matrix)
    # ... use Phonopy features ...
```

---

## FDF Block Parameters (Advanced)

When you need to specify FDF block parameters (like custom phonon band paths), use the `"%block ParamName"` syntax **directly** in `user_params`.

**IMPORTANT**: DO NOT wrap block parameters in `fdf_arguments` - this is deprecated!

### Correct Usage

```python
# ✅ CORRECT: Block parameters directly in user_params
from atomate2.siesta.jobs.phonon import SiestaPhononMaker

maker = SiestaPhononMaker(
    user_params={
        "MD.TypeOfRun": "phonon",
        "MD.FCDispl": "0.01 Ang",
        "a2s_kpts": [8, 8, 8],

        # BandLines block for phonon dispersion
        "%block BandLines": [
            "1  0.0 0.0 0.0  L  # Gamma",
            "20 0.5 0.0 0.5  X",
            "20 0.5 0.25 0.75 W",
            "1  0.5 0.5 0.5  # L",
        ],
    },
    dry_run=True
)
```

### Incorrect Usage (Deprecated)

```python
# ❌ WRONG: Don't nest in fdf_arguments!
maker = SiestaPhononMaker(
    user_params={
        "fdf_arguments": {  # <-- This doesn't work!
            "BandLines": [...]
        }
    }
)
```

### Common Block Parameters for Phonons

- `"%block BandLines"` - K-path for phonon dispersion
- `"%block Geometry.Constraints"` - Fix atoms during phonon calculation

For comprehensive examples, see [02-fdf-block-inputs](../../03-advanced-features/02-fdf-block-inputs/).

---

## Tips for Success

✅ **Relax structure first**: Tight convergence (0.01 eV/Å)
✅ **Tight electronic convergence**: DM.Tolerance = 1e-6
✅ **Dense k-points**: 8×8×8 or higher
✅ **Check sum rule**: Verify force constants satisfy acoustic sum rule
✅ **Compare with Phonopy**: Validate results
⚠️  **Consider PhonopyMaker**: Usually better choice for production
✅ **Block parameters**: Use `"%block ParamName"` directly in `user_params` - NO `fdf_arguments` wrapper!

---

## Next Steps

After native SIESTA phonons:

1. **Try PhonopyMaker**: [07-PhonopyMaker](../07-PhonopyMaker/) - Compare approaches
2. **Phonon workflows**: [02-workflows/06-vibrational-properties](../../02-workflows/06-vibrational-properties/)
3. **QHA**: [02-workflows/06-vibrational-properties/03-qha](../../02-workflows/06-vibrational-properties/03-SiestaQhaFlowMaker/)
4. **Grüneisen**: [02-workflows/06-vibrational-properties/02-gruneisen](../../02-workflows/06-vibrational-properties/02-SiestaGruneisenFlowMaker/)

---

## References

- **SIESTA Manual**: MD.TypeOfRun = phonon section
- **Force constants theory**: Dove "Introduction to Lattice Dynamics"
- **Phonopy comparison**: Phonopy documentation
- **SIESTA phonon utilities**: SIESTA Util/ directory

---

*Back to [01-basics](../README.md) | [Main Tutorial Index](../../README.md)*
