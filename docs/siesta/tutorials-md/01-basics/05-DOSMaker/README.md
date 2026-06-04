# Tutorial: Density of States (DOS) with DOSMaker

**Category**: 01-basics/05-DOSMaker
**Difficulty**: Beginner
**Time**: ~2 min (dry-run), ~10-15 min (full calculation)

---

## Overview

This tutorial demonstrates how to calculate the electronic density of states (DOS) using `DOSMaker`. The DOS provides crucial information about electronic structure, showing how many electronic states are available at each energy level.

**What DOS Tells You**:
- **Band gap**: Energy range with zero states (insulators/semiconductors)
- **Metallic character**: Finite DOS at Fermi level = metal
- **Fermi energy**: Chemical potential of electrons
- **Electronic distribution**: How states are distributed across energy

---

## What You'll Learn

- Using `DOSMaker` for DOS calculations
- DOS energy range and broadening parameters
- Analyzing DOS plots and Fermi energy
- Runtime parameter modification with powerups
- Integration with relaxation workflows

---

## Prerequisites

- **Required tutorials**: [01-RelaxMaker](../01-RelaxMaker/)
- **Required knowledge**: Basic understanding of electronic structure
- **SIESTA configuration**: `~/.atomate2siesta.yaml` set up correctly
- **Structure files**: Located in [00-structures](../../00-structures/)

---

## Key Concepts

### What is Density of States (DOS)?

The DOS, $g(E)$, gives the number of electronic states per unit energy at energy $E$:

$$DOS(E) = \sum_k \sum_n \delta(E - E_{n,k})$$

Where:
- $E_{n,k}$ are the eigenvalues at k-point $k$, band $n$
- $\delta$ is broadened to a Gaussian or Lorentzian

**Physical Meaning**:
- High DOS → many states available → high density of electrons possible
- Zero DOS → band gap → insulator/semiconductor
- DOS at Fermi level → electrical conductivity, heat capacity

### DOS vs Band Structure

- **Band Structure**: $E(k)$ - energy as function of k-point
  - Shows dispersion and effective masses
  - Requires high-symmetry path
  - Good for visualizing band gaps and dispersion

- **DOS**: $g(E)$ - number of states as function of energy
  - Shows state density at each energy
  - Requires dense k-point sampling
  - Good for Fermi level analysis and integration

**Use Both**: Band structure for insight, DOS for quantitative analysis

### Important Parameters

```python
user_params = {
    "kpts": [8, 8, 8],           # Dense mesh for accurate DOS
    "DOS.EnergyMin": "-10 eV",   # Energy range minimum
    "DOS.EnergyMax": "+10 eV",   # Energy range maximum
    "DOS.Broaden": "0.1 eV",     # Gaussian broadening
    "DOS.NumberPoints": 1000,    # Energy grid resolution
}
```

---

## Tutorial Files

This directory contains 2 examples:

1. **`DOSMaker_basic_dos.py`** - Simple DOS calculation with default parameters
2. **`DOSMaker_powerups_customization.py`** - Runtime parameter modification using powerups

---

## Quick Start

### Example 1: Basic DOS Calculation

```python
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import DOSMaker
from jobflow import run_locally

# Load structure
structure = Structure.from_file("../../00-structures/Si.cif")

# Create maker and run
maker = DOSMaker(dry_run=True)  # Preview mode
job = maker.make(structure)
results = run_locally(job, create_folders=True)
```

### Example 2: DOS with Custom Parameters

```python
maker = DOSMaker(
    user_params={
        "kpts": [12, 12, 12],        # Denser k-point mesh
        "PAO.BasisSize": "DZP",
        "Mesh.Cutoff": "300 Ry",
        "DOS.EnergyMin": "-15 eV",
        "DOS.EnergyMax": "+15 eV",
        "DOS.Broaden": "0.05 eV",    # Sharper features
    },
    dry_run=True
)
```

### Example 3: DOS After Relaxation

```python
from atomate2.siesta.jobs.core import RelaxMaker

# Two-step workflow: relax → DOS
relax_maker = RelaxMaker.fixed_cell_relaxation()
dos_maker = DOSMaker()

# Create jobs
relax_job = relax_maker.make(structure)
dos_job = dos_maker.make(structure, prev_dir=relax_job.output.dir_name)

# Chain them
from jobflow import Flow
workflow = Flow([relax_job, dos_job])
results = run_locally(workflow, create_folders=True)
```

### Example 4: Using Powerups

```python
from atomate2.siesta.powerups import update_user_siesta_settings

maker = DOSMaker(dry_run=True)
job = maker.make(structure)

# Modify parameters at runtime
job = update_user_siesta_settings(job, {
    "kpts": [10, 10, 10],
    "DOS.Broaden": "0.2 eV",
})

results = run_locally(job, create_folders=True)
```

---

## Run Modes

### 1. Dry-Run Mode

```bash
python DOSMaker_basic_dos.py  # With dry_run=True
```

**Output**:
```
preview_output/job_*/
├── siesta.fdf           # Contains DOS parameters
├── structure.fdf
├── *.psml
└── structure.cif
```

**Check DOS parameters**:
```bash
grep "DOS\." preview_output/job_*/siesta.fdf
```

### 2. Local Execution

```bash
# Edit: Set dry_run=False
python DOSMaker_basic_dos.py
```

**Output**:
```
job_*/
├── siesta.fdf
├── siesta.out
├── siesta.DOS          # Total DOS file
├── siesta.PDOS.xml     # Projected DOS (if enabled)
└── [SIESTA output files]
```

**Time**: ~10-15 minutes for small systems

---

## Expected Output

### DOS File Format

The `siesta.DOS` file contains:
```
# Energy(eV)  Total_DOS(states/eV)  Integrated_DOS(electrons)
-10.000      0.234                  0.012
 -9.900      0.456                  0.034
 -9.800      0.678                  0.067
  ...
```

### Analyzing DOS

```python
import numpy as np
import matplotlib.pyplot as plt

# Read DOS file
data = np.loadtxt("job_*/siesta.DOS")
energy = data[:, 0]
dos = data[:, 1]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(energy, dos, 'b-', linewidth=2)
plt.axvline(x=fermi_energy, color='r', linestyle='--', label='Fermi level')
plt.xlabel('Energy (eV)')
plt.ylabel('DOS (states/eV)')
plt.title('Density of States')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('dos_plot.png', dpi=300)
plt.show()
```

### Extract Fermi Energy

```bash
grep "Fermi" job_*/siesta.out
# Output: "siesta: Ef = -5.3456"
```

Or with Python:
```python
from pymatgen.io.siesta import SiestaOutput

output = SiestaOutput("job_*/siesta.out")
fermi = output.fermi_energy
print(f"Fermi energy: {fermi:.4f} eV")
```

### Determine Band Gap

```python
def find_band_gap(energy, dos, fermi, threshold=0.01):
    """Find band gap from DOS."""
    # States below/above Fermi
    below_fermi = dos[energy < fermi]
    above_fermi = dos[energy > fermi]

    # Check if metallic (DOS at Fermi > threshold)
    dos_at_fermi = np.interp(fermi, energy, dos)
    if dos_at_fermi > threshold:
        return 0.0, "Metallic"

    # Find VBM and CBM
    vbm_energy = energy[energy < fermi][-1]
    cbm_energy = energy[energy > fermi][0]
    gap = cbm_energy - vbm_energy

    return gap, "Semiconductor/Insulator"

gap, character = find_band_gap(energy, dos, fermi)
print(f"Band gap: {gap:.3f} eV ({character})")
```

---

## Common Issues

### Issue 1: "DOS is too noisy"

**Cause**: Insufficient k-points or too narrow broadening

**Solutions**:
```python
# Increase k-point sampling
user_params={"kpts": [12, 12, 12]}  # or denser

# Increase broadening
user_params={"DOS.Broaden": "0.2 eV"}  # Smoother but less detail
```

### Issue 2: "DOS has strange features"

**Cause**: Unconverged electronic structure

**Solution**: Check SCF convergence
```bash
grep "SCF Convergence" job_*/siesta.out
# Should show "SCF Converged"
```

If not converged:
```python
user_params={
    "DM.Tolerance": "1e-5",     # Tighter convergence
    "DM.NumberPulay": 8,        # More history
}
```

### Issue 3: "Band gap incorrect"

**Cause**: GGA functionals underestimate band gaps

**Solutions**:
1. **Accept limitation**: GGA typically underestimates by 30-50%
2. **Use DFT+U**: For strongly correlated systems (see [03-advanced-features/08-dftu](../../03-advanced-features/08-dftu/))
3. **Compare trends**: Relative gaps are more reliable than absolute values

### Issue 4: "Fermi level position wrong"

**Cause**: Wrong number of electrons or charge

**Solution**: Check electron count
```bash
grep "Number of electrons" job_*/siesta.out
```

For charged systems:
```python
user_params={"NetCharge": "+1"}  # +1 for cation, -1 for anion
```

### Issue 5: "Unknown FDF parameter: fdf_arguments"

**Error**: `ValueError: Unknown FDF parameter(s): fdf_arguments`

**Cause**: Using deprecated `fdf_arguments` wrapper syntax

**Solution**:
```python
# ❌ OLD (doesn't work):
user_params = {
    "fdf_arguments": {
        "ProjectedDensityOfStates": [...]
    }
}

# ✅ NEW (correct):
user_params = {
    "%block ProjectedDensityOfStates": [...]  # Directly in user_params!
}
```

**Note**: Block parameters should be specified **directly** in `user_params`, NOT nested in `fdf_arguments`. See [FDF Block Parameters](#fdf-block-parameters-advanced) section below.

---

## DOS Parameters Reference

### Energy Range

```python
"DOS.EnergyMin": "-10 eV"    # Start energy (relative to Fermi)
"DOS.EnergyMax": "+10 eV"    # End energy
```

**Tips**:
- Too narrow: Miss important features
- Too wide: Waste computation
- Typical: ±10 eV covers valence + conduction bands

### Broadening

```python
"DOS.Broaden": "0.1 eV"      # Gaussian broadening width
```

**Effects**:
- **Small (0.01-0.05 eV)**: Sharp features, noisy
- **Medium (0.1-0.2 eV)**: Good balance
- **Large (0.3-0.5 eV)**: Smooth, loss of detail

**Physical meaning**: Simulates finite lifetime of states, experimental resolution

### Energy Grid

```python
"DOS.NumberPoints": 1000     # Points in energy grid
```

**Guidelines**:
- Minimum: 500 points
- Standard: 1000 points
- High resolution: 2000+ points

---

## FDF Block Parameters (Advanced)

When you need to specify FDF block parameters (like custom DOS energy ranges), use the `"%block ParamName"` syntax **directly** in `user_params`.

**IMPORTANT**: DO NOT wrap block parameters in `fdf_arguments` - this is deprecated!

### Correct Usage

```python
# ✅ CORRECT: Block parameters directly in user_params
from atomate2.siesta.jobs.core import DOSMaker

maker = DOSMaker(
    user_params={
        "a2s_kpts": [8, 8, 8],
        "Mesh.Cutoff": "300 Ry",

        # ProjectedDensityOfStates block (custom DOS parameters)
        "%block ProjectedDensityOfStates": [
            "EF -15.0 15.0 0.05 600 eV",  # Energy range: -15 to +15 eV
        ],
    },
    dry_run=True
)
```

### Incorrect Usage (Deprecated)

```python
# ❌ WRONG: Don't nest in fdf_arguments!
maker = DOSMaker(
    user_params={
        "fdf_arguments": {  # <-- This doesn't work!
            "ProjectedDensityOfStates": [...]
        }
    }
)
```

### Common Block Parameters for DOS

- `"%block ProjectedDensityOfStates"` - Custom DOS energy range and resolution
- `"%block DOS.kgrid.MonkhorstPack"` - Custom k-point grid for DOS sampling
- `"%block PDOS.kgrid.MonkhorstPack"` - K-point grid for projected DOS

For comprehensive examples, see [02-fdf-block-inputs](../../03-advanced-features/02-fdf-block-inputs/).

---

## Alternative: Using CLI Tool

Generate DOS scripts automatically:

```bash
# Interactive mode
atomate2siesta-maker --interactive
# Select: "dos" → Choose structure → Done!

# Command-line mode
atomate2siesta-maker dos Si.cif
```

---

## Advanced Customization

### Using Tier Presets

```python
from atomate2.siesta.sets.tiers import apply_tier_preset

maker = DOSMaker()
maker = apply_tier_preset(maker, "band_structure")  # For electronic structure and DOS
```

### Using Recipe Book

```python
from atomate2.siesta.recipes import RecipeBook

# One-line DOS workflow (relax + DOS)
flow = RecipeBook.dos_workflow(structure)
```

### Projected DOS (PDOS)

For atom/orbital-resolved DOS, use `PDOSMaker` (see [06-PDOSMaker](../06-PDOSMaker/))

---

## Tips for Success

✅ **Use dense k-points**: 8×8×8 minimum, 12×12×12 better
✅ **Check convergence**: Verify SCF converged in siesta.out
✅ **Plot your DOS**: Visual inspection catches errors
✅ **Compare with bands**: DOS should match band structure features
✅ **Note GGA limitations**: Band gaps will be underestimated
✅ **Block parameters**: Use `"%block ParamName"` directly in `user_params` - NO `fdf_arguments` wrapper!

---

## Next Steps

After completing DOS tutorials:

1. **Projected DOS**: [06-PDOSMaker](../06-PDOSMaker/) - Orbital and atom decomposition
2. **Band structure**: [02-BandStructureMaker](../02-BandStructureMaker/) - Complementary view
3. **Optical properties**: [03-advanced-features/07-optical-properties](../../03-advanced-features/07-optical-properties/)
4. **One-line workflows**: [03-advanced-features/08-recipe-book](../../03-advanced-features/08-recipe-book/)

---

## References

- **SIESTA Manual**: DOS chapter
- **Ashcroft & Mermin**: "Solid State Physics" - DOS theory
- **Pymatgen DOS**: [Documentation](https://pymatgen.org/pymatgen.electronic_structure.dos.html)

---

*Back to [01-basics](../README.md) | [Main Tutorial Index](../../README.md)*
