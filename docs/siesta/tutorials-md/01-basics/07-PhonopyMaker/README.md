# Tutorial: Phonon Calculations with PhonopyMaker

**Category**: 01-basics/07-PhonopyMaker
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~1-2 hours (full calculation)

---

## Overview

This tutorial introduces phonon calculations using `PhonopyMaker`, which integrates SIESTA with Phonopy for phonon band structure, density of states, and thermal properties. Phonons are quantized lattice vibrations that govern thermal, mechanical, and transport properties of materials.

**What Phonons Tell You**:
- **Structural stability**: Imaginary modes → unstable structure
- **Thermal properties**: Heat capacity, Debye temperature, entropy
- **Sound velocities**: Acoustic mode slopes
- **Thermodynamics**: Free energy, thermal expansion (via QHA)

---

## What You'll Learn

- Using `PhonopyMaker` for phonon calculations
- Supercell generation and finite displacement method
- Phonon band structure and DOS analysis
- Automatic phonon plotting
- Thermal property calculations
- Integration with QHA workflows

---

## Prerequisites

- **Required tutorials**: [01-RelaxMaker](../01-RelaxMaker/)
- **Required knowledge**: Basic lattice dynamics, phonon concepts
- **Required software**:
  - SIESTA installed and configured
  - Phonopy installed: `pip install phonopy`
- **Structure files**: Located in [00-structures](../../00-structures/)

---

## Key Concepts

### Phonon Theory Basics

**Phonon**: Quantized lattice vibration with:
- Frequency $\omega(q)$ at wavevector $q$
- Dispersion relation: $\omega = \omega(q)$

**Dynamical Matrix**: $D_{\alpha\beta}(q) = \frac{1}{\sqrt{M_\alpha M_\beta}} \Phi_{\alpha\beta}(q)$
- $\Phi$: Force constant matrix
- $M$: Atomic masses

**Phonon Frequencies**: Eigenvalues of dynamical matrix

### Finite Displacement Method

Phonopy uses supercell approach:

1. **Create supercell**: Typically 2×2×2 or larger
2. **Displace atoms**: Small displacements (0.01 Å) in symmetry-reduced set
3. **Calculate forces**: SIESTA computes forces on all atoms
4. **Extract force constants**: From force-displacement relationship
5. **Solve dynamical matrix**: Get phonon frequencies

**Supercell Size Requirements**:
- Minimum: 2×2×2 (for simple systems)
- Recommended: Ensure all cell dimensions > 15 Å
- Larger supercells: Better long-range interactions, more expensive

### Output Files

```
job_*/
├── phonopy.yaml          # Phonopy input/output
├── FORCE_CONSTANTS       # Harmonic force constants
├── band.yaml             # Phonon band structure data
├── dos.dat               # Phonon DOS
├── thermal_properties.yaml  # Cv, S, F vs T
├── phonon_bands.png      # AUTO-GENERATED PLOT
├── phonon_dos.png        # AUTO-GENERATED PLOT
└── thermal_properties.png   # AUTO-GENERATED PLOT
```

---

## Tutorial Files

This directory contains:

1. **`01_basic.py`** - Simple phonon calculation with automatic plotting
2. **`01_basic/`** - Subdirectory with additional examples

---

## Quick Start

### Example 1: Basic Phonon Calculation

```python
from pymatgen.core import Structure
from atomate2.siesta.jobs.phonon import PhonopyMaker
from jobflow import run_locally

# Load structure (MUST be relaxed first!)
structure = Structure.from_file("../../00-structures/Si.cif")

# Create maker with supercell specification
maker = PhonopyMaker(
    min_length=15.0,  # Auto-generate supercell with min dimension 15 Å
    dry_run=True
)

# Generate and run
job = maker.make(structure)
results = run_locally(job, create_folders=True)
```

**Output**: Generates 3 automatic plots (bands, DOS, thermal properties)

### Example 2: Explicit Supercell Matrix

```python
import numpy as np

maker = PhonopyMaker(
    supercell_matrix=np.diag([2, 2, 2]),  # 2×2×2 supercell
    displacement=0.01,  # Displacement magnitude (Å)
    dry_run=True
)
```

### Example 3: Custom SIESTA Parameters

```python
from atomate2.siesta.jobs.core import StaticMaker

# Customize force calculation settings
static_maker = StaticMaker(
    user_params={
        "PAO.BasisSize": "DZP",
        "kpts": [6, 6, 6],
        "Mesh.Cutoff": "300 Ry",
        "MD.MaxForceTol": "0.01 eV/Ang",  # Tight for accurate forces
    }
)

maker = PhonopyMaker(
    min_length=15.0,
    bulk_relax_maker=None,  # Use pre-relaxed structure
    static_energy_maker=static_maker,  # Custom force calculations
    dry_run=True
)
```

### Example 4: With Relaxation First

```python
from atomate2.siesta.jobs.core import RelaxMaker

# Automatic relaxation before phonons
relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "PAO.BasisSize": "DZP",
        "kpts": [8, 8, 8],
        "MD.MaxForceTol": "0.01 eV/Ang",  # Tight!
    }
)

maker = PhonopyMaker(
    min_length=15.0,
    bulk_relax_maker=relax_maker,  # Enable relaxation
    dry_run=True
)
```

---

## Run Modes

### 1. Dry-Run Mode

```bash
python 01_basic.py  # With dry_run=True
```

**What happens**:
- Creates supercell
- Generates displaced structures
- Shows how many force calculations needed
- No SIESTA execution

**Output**:
```
preview_output/
├── job_relax_*/               # (if relaxation enabled)
├── job_phonon_displacements_*/  # Displaced structures
└── job_phonon_analysis_*/     # Analysis job (no plots in dry-run)
```

### 2. Local Execution

```bash
# Edit: Set dry_run=False
python 01_basic.py
```

**What happens**:
1. Relaxes structure (if `bulk_relax_maker` set)
2. Creates supercell
3. Runs ~20-50 force calculations (depending on symmetry)
4. Extracts force constants
5. Computes phonon bands and DOS
6. **Automatically generates 3 plots**

**Time**: 1-2 hours for Si (depends on supercell size and system)

---

## Expected Output

### Automatic Plots (NEW!)

PhonopyMaker automatically generates 3 publication-quality plots:

#### 1. `phonon_bands.png`
- Phonon dispersion along high-symmetry path
- Acoustic modes (3 branches starting at 0)
- Optical modes (higher frequencies)
- Γ, X, W, L, K points labeled

#### 2. `phonon_dos.png`
- Phonon density of states
- Peaks correspond to van Hove singularities
- Area under curve = 3N modes (N = number of atoms)

#### 3. `thermal_properties.png`
- Heat capacity Cv(T)
- Entropy S(T)
- Free energy F(T)
- Temperature range: 0-1000 K

### Analyzing Phonon Results

```python
import numpy as np
import matplotlib.pyplot as plt
from phonopy import load

# Load phonopy results
phonon = load("job_*/phonopy.yaml")

# Check for imaginary modes (instability)
qpoints, freqs, _ = phonon.get_band_structure_dict()
min_freq = freqs.min()
if min_freq < 0:
    print(f"WARNING: Imaginary modes detected! Min freq: {min_freq:.2f} THz")
    print("Structure may be unstable or insufficiently relaxed.")
else:
    print(f"All phonon modes are real. Min freq: {min_freq:.2f} THz")

# Extract thermal properties
tp = phonon.get_thermal_properties_dict(t_step=10, t_max=1000)
temps = tp['temperatures']
cv = tp['heat_capacity']  # J/K/mol
entropy = tp['entropy']   # J/K/mol
free_energy = tp['free_energy']  # kJ/mol

# Debye temperature (approximate from heat capacity)
debye_temp = temps[np.argmax(cv > 0.95 * 3 * 8.314)]  # 3R for Dulong-Petit
print(f"Approximate Debye temperature: {debye_temp:.0f} K")
```

### Zero-Point Energy

```python
# Calculate zero-point energy
from phonopy.units import VaspToTHz, THzToEv

zpe = 0
for freq in phonon.get_mesh_dict(mesh=[20, 20, 20])['frequencies'].flatten():
    if freq > 0:  # Skip imaginary modes
        zpe += 0.5 * freq * THzToEv * VaspToTHz

zpe_per_atom = zpe / len(structure)
print(f"Zero-point energy: {zpe:.4f} eV ({zpe_per_atom:.4f} eV/atom)")
```

---

## Common Issues

### Issue 1: "Imaginary modes at Γ"

**Symptoms**: Negative frequencies at Γ point

**Causes & Solutions**:

1. **Structure not fully relaxed**:
   ```python
   # Use tighter convergence
   relax_maker = RelaxMaker.fixed_cell_relaxation(
       user_params={"MD.MaxForceTol": "0.005 eV/Ang"}  # Very tight!
   )
   ```

2. **Force calculation not converged**:
   ```python
   static_maker = StaticMaker(
       user_params={
           "kpts": [10, 10, 10],      # Denser
           "Mesh.Cutoff": "400 Ry",   # Higher
           "DM.Tolerance": "1e-6",    # Tighter
       }
   )
   ```

3. **Supercell too small**:
   ```python
   maker = PhonopyMaker(min_length=20.0)  # Increase from 15 Å
   ```

### Issue 2: "Forces inconsistent between displacements"

**Symptoms**: Phonopy warnings about force consistency

**Solution**: Increase SCF convergence
```python
user_params={
    "DM.Tolerance": "1e-6",
    "DM.NumberPulay": 8,
    "DM.MixingWeight": 0.1,  # Slower mixing, more stable
}
```

### Issue 3: "Phonon calculation takes too long"

**Causes**:
- Large supercell (many atoms)
- Many displacement patterns

**Solutions**:
1. **Use symmetry** (Phonopy does this automatically)
2. **Smaller supercell** (but check convergence!)
   ```python
   maker = PhonopyMaker(
       supercell_matrix=np.diag([2, 2, 2])  # Instead of 3×3×3
   )
   ```
3. **Submit to HPC** (see [03-advanced-features/03-infrastructure](../../03-advanced-features/03-infrastructure/))

### Issue 4: "Plots not generated"

**Cause**: Dry-run mode or analysis job failed

**Solution**:
1. Check `dry_run=False`
2. Verify analysis job completed:
   ```bash
   ls job_phonon_analysis_*/phonon_*.png
   ```
3. Check for errors in analysis job output

---

## Phonon Analysis Best Practices

### 1. Always Relax First

```python
# ❌ WRONG - Using unrelaxed structure
structure = Structure.from_file("structure.cif")
maker = PhonopyMaker(bulk_relax_maker=None)  # Skips relaxation

# ✅ CORRECT - Include relaxation
maker = PhonopyMaker(
    bulk_relax_maker=RelaxMaker.fixed_cell_relaxation(
        user_params={"MD.MaxForceTol": "0.01 eV/Ang"}
    )
)
```

### 2. Use Tight Force Convergence

For accurate phonons:
```python
static_maker = StaticMaker(
    user_params={
        "MD.MaxForceTol": "0.005 eV/Ang",  # 5× tighter than default
        "DM.Tolerance": "1e-6",
    }
)
```

### 3. Check Supercell Convergence

Test phonon DOS with different supercell sizes:
```python
for size in [2, 3, 4]:
    maker = PhonopyMaker(
        supercell_matrix=np.diag([size, size, size])
    )
    # Compare DOS - should converge by 3×3×3
```

### 4. Validate with Experiments

Compare calculated phonon frequencies with:
- Raman spectroscopy
- Infrared spectroscopy
- Inelastic neutron scattering
- Acoustic measurements (sound velocities from slope of acoustic branches)

---

## Advanced Features

### Phonon Calculation Workflow

For more control, see the full workflow maker in [02-workflows/06-vibrational-properties](../../02-workflows/06-vibrational-properties/):

```python
from atomate2.siesta.flows.phonon import PhonopyFlowMaker

# Complete phonon workflow with automatic plotting
flow_maker = PhonopyFlowMaker(
    min_length=15.0,
    relax_maker_kwargs={"user_params": {"MD.MaxForceTol": "0.01 eV/Ang"}},
    generate_frequencies_eigenvectors_kwargs={"write_json": True},
)

flow = flow_maker.make(structure)
```

### Grüneisen Parameters and QHA

For thermal expansion and temperature-dependent properties:

See [02-workflows/06-vibrational-properties/02-gruneisen](../../02-workflows/06-vibrational-properties/02-SiestaGruneisenFlowMaker/) and [03-qha](../../02-workflows/06-vibrational-properties/03-SiestaQhaFlowMaker/)

---

## Tips for Success

✅ **Relax structure first**: Use tight convergence (0.01 eV/Å forces)
✅ **Check for imaginary modes**: Negative frequencies = problem
✅ **Use adequate supercell**: min_length ≥ 15 Å
✅ **Tight force calculations**: DM.Tolerance = 1e-6
✅ **Verify plots generated**: Check all 3 PNG files created
✅ **Compare with experiments**: Validate frequencies and thermal properties

---

## Comparison: PhonopyMaker vs SiestaPhononMaker

| Feature | PhonopyMaker | SiestaPhononMaker |
|---------|--------------|-------------------|
| Method | Finite displacements | Force constants matrix |
| Software | Phonopy + SIESTA | SIESTA native |
| Flexibility | High (Phonopy features) | Limited |
| Automatic plots | ✓ Yes (3 plots) | Depends |
| QHA/Grüneisen | ✓ Easy integration | Manual |
| Computational cost | ~20-50 calculations | ~10-20 calculations |
| **Recommendation** | **Default choice** | Special cases only |

See [08-SiestaPhononMaker](../08-SiestaPhononMaker/) for native SIESTA phonon approach.

---

## Next Steps

After completing phonon tutorials:

1. **Grüneisen parameters**: [02-workflows/06-vibrational-properties/02-gruneisen](../../02-workflows/06-vibrational-properties/02-SiestaGruneisenFlowMaker/)
2. **Quasi-harmonic approximation (QHA)**: [02-workflows/06-vibrational-properties/03-qha](../../02-workflows/06-vibrational-properties/03-SiestaQhaFlowMaker/)
3. **Thermal expansion**: Combine phonons with EOS
4. **Native SIESTA phonons**: [08-SiestaPhononMaker](../08-SiestaPhononMaker/)

---

## References

- **Phonopy Documentation**: https://phonopy.github.io/phonopy/
- **SIESTA-Phonopy Interface**: SIESTA manual phonon chapter
- **Lattice Dynamics**: Dove "Introduction to Lattice Dynamics"
- **Thermal Properties**: Ashcroft & Mermin "Solid State Physics"

---

*Back to [01-basics](../README.md) | [Main Tutorial Index](../../README.md)*
