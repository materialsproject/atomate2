# Tutorial: Band Structure Calculation with BandStructureMaker

**Category**: 01-basics/02-BandStructureMaker
**Difficulty**: Beginner
**Time**: ~2 min (dry-run), ~15-20 min (full calculation)

---

## Overview

This tutorial demonstrates how to calculate electronic band structure using the `BandStructureMaker`. Band structure calculations are essential for understanding the electronic properties of materials, including:

- Band gap determination (semiconductors/insulators)
- Metallic vs insulating character
- Electronic dispersion and effective masses
- Comparison with experimental spectroscopy

---

## What You'll Learn

- How to use `BandStructureMaker` for automatic workflows
- Automatic high-symmetry k-path generation
- Multi-step job creation (SCF + band structure)
- Parameter customization with `user_params`
- Runtime modifications with powerups
- Electronic structure analysis basics

---

## Prerequisites

- **Required tutorials**: [01-RelaxMaker](../01-RelaxMaker/)
- **Required knowledge**: Basic understanding of electronic band structure
- **SIESTA configuration**: `~/.atomate2siesta.yaml` set up correctly
- **Structure files**: Located in [00-structures](../../00-structures/)

---

## Key Concepts

### Band Structure Calculation

Band structure calculations involve two steps:

1. **SCF calculation**: Self-consistent field calculation to obtain converged charge density
2. **Band structure calculation**: Non-self-consistent calculation along high-symmetry k-path

The `BandStructureMaker` handles both steps automatically!

### High-Symmetry K-Path

The k-path is automatically generated based on crystal symmetry:
- **Cubic (FCC)**: Γ → X → W → K → Γ → L
- **Cubic (BCC)**: Γ → H → N → Γ → P → H
- **Hexagonal**: Γ → M → K → Γ → A → L → H → A

For Silicon (FCC), the path is: Γ → X → U|K → Γ → L → W → X

### Important Parameters

- **PAO.BasisSize**: Basis set quality (DZP recommended for band structure)
- **Mesh.Cutoff**: Real-space grid cutoff (200-300 Ry typical)
- **k-points**: Dense mesh for SCF, special path for bands

---

## Tutorials in This Section

1. **`BandStructureMaker_basic.py`**: Simple band structure calculation (default parameters)
2. **`BandStructureMaker_custom_params.py`**: Customize parameters using `user_params` in maker
3. **`BandStructureMaker_powerups.py`**: Runtime modifications using powerups functions

---

## Alternative: Using CLI Tool

Instead of writing Python code, generate this band structure script automatically:

```bash
# Quick generation
atomate2siesta-maker bands Si.cif

# Or use interactive mode (zero memorization!)
atomate2siesta-maker --interactive
```

The CLI tool generates a ready-to-run Python script with the same structure as this tutorial.

**Benefits**:
- No Python knowledge required
- Step-by-step prompts in interactive mode
- Automatic preset detection
- Choice of execution mode (local/remote/dry-run)

See the [CLI Tools documentation](../../../docs/source/cli-tools.rst) for all 13 available workflows.

---

## Run Modes

### 1. Dry-Run (Preview)

```bash
# Edit tutorial.py: Set dry_run=True
python tutorial.py
```

**What it does**:
- Creates folder structure
- Shows multi-job workflow (SCF + bands)
- No SIESTA execution

**Use for**:
- Verifying workflow setup
- Checking job dependencies
- Previewing folder structure

### 2. Local Execution

```bash
python tutorial.py  # With dry_run=False
```

**What it does**:
- Runs SCF calculation (~10 min)
- Runs band structure calculation (~5-10 min)
- Generates electronic structure data

**Requirements**:
- SIESTA installed locally
- Sufficient RAM (~2-4 GB for Si)

### 3. HPC Submission

```bash
# Use jobflow-remote
from jobflow_remote import submit_flow
submit_flow(job, project="production")
```

**What it does**:
- Submits workflow to HPC cluster
- Jobs run when resources available
- Retrieves results automatically

**Requirements**:
- jobflow-remote configured (see [03-infrastructure/02-job-submission](../../03-advanced-features/03-infrastructure/02-job-submission/))
- HPC cluster access

---

## Expected Output

### Dry-Run Mode
```
preview_output/
├── job_scf_*/
│   ├── siesta.fdf
│   ├── structure.fdf
│   └── pseudopotentials/
└── job_bands_*/
    ├── siesta.fdf
    ├── structure.fdf
    └── pseudopotentials/
```

### Local/Submit Mode
```
job_scf_*/
├── siesta.fdf
├── siesta.out
├── Si.bands
└── [SIESTA output files]

job_bands_*/
├── siesta.fdf
├── siesta.out
├── Si.bands       # Band structure data
└── Si.bands.gp    # Gnuplot script
```

---

## Configuration Options

### Basis Set Quality

```python
user_params = {"PAO.BasisSize": "DZP"}  # Options: SZ, DZ, SZP, DZP, TZP
```

- **SZ/DZ**: Fast but less accurate
- **DZP**: Good balance (recommended)
- **TZP**: High accuracy but slower

### Mesh Cutoff

```python
user_params = {"Mesh.Cutoff": "200 Ry"}
```

- **150 Ry**: Quick tests
- **200-300 Ry**: Standard
- **400+ Ry**: High accuracy

---

## Analysis and Visualization

### Using Pymatgen

```python
from pymatgen.io.siesta import SiestaOutput

# Read band structure
output = SiestaOutput("job_bands_*/siesta.out")
bs = output.get_band_structure()

# Plot
from pymatgen.electronic_structure.plotter import BSPlotter
plotter = BSPlotter(bs)
plotter.get_plot().show()
```

### Key Properties to Extract

1. **Band gap**: Energy difference between VBM and CBM
2. **Direct vs indirect**: Compare VBM and CBM k-points
3. **Effective masses**: Curvature near band edges
4. **Density of states**: Electronic DOS at Fermi level

---

## Common Issues

### Issue 1: "SCF not converged"

**Cause**: Insufficient k-points or problematic starting density

**Solution**:
```python
# Add denser k-point mesh
from atomate2.siesta.powerups import update_user_siesta_settings
job = update_user_siesta_settings(job, {
    "a2s_kpts": [8, 8, 8],  # Denser mesh for SCF
})
```

### Issue 2: "Band structure looks strange"

**Cause**: Unconverged parameters or poor basis set

**Solution**:
- Increase `Mesh.Cutoff` to 300-400 Ry
- Use `PAO.BasisSize = "TZP"` for better accuracy
- Check convergence (see [02-workflows/01-convergence](../../02-workflows/01-convergence/))

### Issue 3: "Job takes too long"

**Cause**: Dense k-point mesh or large system

**Solution**:
- Use dry-run to preview
- Submit to HPC cluster instead of local
- Reduce mesh cutoff for testing (increase for production)

### Issue 4: "Unknown FDF parameter: fdf_arguments"

**Error**: `ValueError: Unknown FDF parameter(s): fdf_arguments`

**Cause**: Using deprecated `fdf_arguments` wrapper syntax

**Solution**:
```python
# ❌ OLD (doesn't work):
user_params = {
    "fdf_arguments": {
        "PAO.Basis": [...]
    }
}

# ✅ NEW (correct):
user_params = {
    "%block PAO.Basis": [...]  # Directly in user_params!
}
```

**Note**: Block parameters should be specified **directly** in `user_params`, NOT nested in `fdf_arguments`. See [FDF Block Parameters](#fdf-block-parameters-advanced) section below.

---

## Validation

### Compare with Known Results

For Silicon:
- **Band gap**: ~0.6 eV (GGA underestimates, experiment ~1.1 eV)
- **Indirect gap**: Γ → X valley transition
- **Valence band maximum**: At Γ point

### Experimental Comparison

- **Angle-resolved photoemission spectroscopy (ARPES)**: Band dispersion
- **Optical absorption**: Band gap
- **Calculated vs experimental**: Expect GGA band gap underestimation

---

## Advanced Customization

For more sophisticated parameter control and workflow customization:

**📖 Parameter Customization Methods**
- [Makers vs FlowMakers](../../../docs/source/makers-vs-flowmakers.rst) - Comprehensive guide on when to use `user_params`, tier presets, or powerups
- [Powerups System](../../../docs/source/features.rst#powerups-system) - Runtime parameter modifications for jobs and flows
- [Tier System](../../../docs/source/tier-system.rst) - Material-specific parameter presets

**🎯 Quick Examples**

*Using tier presets:*
```python
from atomate2.siesta.sets.tiers import apply_tier_preset
from atomate2.siesta.jobs.core import BandStructureMaker

maker = BandStructureMaker()
maker = apply_tier_preset(maker, "band_structure")
```

*Using powerups for runtime modifications:*
```python
from atomate2.siesta.powerups import update_user_siesta_settings

job = maker.make(structure)
job = update_user_siesta_settings(job, {
    "Mesh.Cutoff": "300 Ry",
    "PAO.BasisSize": "TZP",
})
```

*Using Recipe Book (one-liner):*
```python
from atomate2.siesta.recipes import RecipeBook

# Complete band structure workflow in one line
flow = RecipeBook.band_structure_workflow(structure)
```

---

## FDF Block Parameters (Advanced)

When you need to specify FDF block parameters (like custom basis sets, constraints, or DFT+U), use the `"%block ParamName"` syntax **directly** in `user_params`.

**IMPORTANT**: DO NOT wrap block parameters in `fdf_arguments` - this is deprecated!

### Correct Usage

```python
# ✅ CORRECT: Block parameters directly in user_params
from atomate2.siesta.jobs.core import BandStructureMaker

maker = BandStructureMaker(
    user_params={
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [6, 6, 6],

        # Custom basis block (if needed)
        "%block PAO.Basis": [
            "Si 2",
            " n=3 0 2 P 1",
            "   4.5 0.0",
            " n=3 1 1 P 1",
            "   5.0 0.0",
        ],
    }
)
```

### Incorrect Usage (Deprecated)

```python
# ❌ WRONG: Don't nest in fdf_arguments!
maker = BandStructureMaker(
    user_params={
        "fdf_arguments": {  # <-- This doesn't work!
            "PAO.Basis": [...]
        }
    }
)
```

### Common Block Parameters

- `"%block PAO.Basis"` - Custom orbital definitions
- `"%block DM.InitSpin"` - Initial magnetic moments (for spin-polarized)
- `"%block Geometry.Constraints"` - Fix atoms during relaxation
- `"%block DFTU.Proj"` - DFT+U projectors

For comprehensive examples, see:
- [02-fdf-block-inputs](../../03-advanced-features/02-fdf-block-inputs/)
- [07-basis-set-customization](../../03-advanced-features/07-basis-set-customization/)

---

## Tips for Success

✅ **Always start with dry-run**: Catch mistakes early
✅ **Check SCF convergence**: Verify SCF job completed successfully before analyzing bands
✅ **Use adequate k-points**: Dense mesh for SCF, automatic k-path for bands
✅ **Visualize results**: Plot band structure to verify calculations
✅ **Compare with literature**: Validate band gap and dispersion

---

## Next Steps

After completing this tutorial:

1. **Explore convergence**: [02-workflows/01-convergence](../../02-workflows/01-convergence/)
2. **Learn DOS calculations**: [01-basics/05-DOSMaker](../05-DOSMaker/) and [01-basics/06-PDOSMaker](../06-PDOSMaker/)
3. **Try Recipe Book**: [03-advanced-features/08-recipe-book](../../03-advanced-features/08-recipe-book/) for one-line workflows
4. **Build workflows**: [04-RelaxMaker-StaticMaker](../04-RelaxMaker-StaticMaker/) for multi-step calculations

---

## References

- SIESTA Band Structure: [SIESTA Manual - Electronic Structure](https://siesta-project.org/SIESTA_MATERIAL/Docs/Manuals/)
- Pymatgen Band Structure: [Pymatgen Documentation](https://pymatgen.org/pymatgen.electronic_structure.bandstructure.html)

---

*Back to [01-basics](../README.md) | [Main Tutorial Index](../../README.md)*
