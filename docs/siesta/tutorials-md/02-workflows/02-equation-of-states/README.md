# Tutorial: Equation of State (EOS) Workflows

**Category**: 02-workflows/02-equation-of-states
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~30-60 min (full workflow)

---

## Overview

This tutorial demonstrates equation of state (EOS) calculations using various FlowMakers. The EOS describes how a material's energy varies with volume, providing crucial information about bulk modulus, equilibrium volume, and pressure-volume relationships.

**What EOS Calculations Tell You**:
- **Bulk modulus (B₀)**: Resistance to compression
- **Equilibrium volume (V₀)**: Minimum energy volume
- **Pressure derivative (B'₀)**: How bulk modulus changes with pressure
- **Cohesive energy**: Binding energy of the crystal

---

## What You'll Learn

- Using `SiestaEosFlowMaker` for basic EOS workflows
- EOS fitting with Birch-Murnaghan equation
- Combining EOS with basis set convergence
- Automatic EOS plotting
- Analyzing bulk modulus and equilibrium lattice parameters

---

## Prerequisites

- **Required tutorials**: [01-basics/01-RelaxMaker](../../01-basics/01-RelaxMaker/)
- **Required knowledge**: Basic thermodynamics, bulk modulus concept
- **Recommended**: [01-convergence](../01-convergence/) - Converged parameters
- **Structure files**: Located in [00-structures](../../00-structures/)

---

## Key Concepts

### Equation of State

The EOS relates energy E to volume V:

**Birch-Murnaghan EOS (3rd order)**:

$$
E(V) = E_0 + \frac{B_0 V}{B'_0} \left[\frac{(V_0/V)^{B'_0}}{B'_0 - 1} + 1\right] - \frac{B_0 V_0}{B'_0 - 1}
$$

Where:
- $E_0$: Energy at equilibrium
- $V_0$: Equilibrium volume
- $B_0$: Bulk modulus at V₀
- $B'_0$: Pressure derivative of bulk modulus

**Physical Meaning**:
- $B_0$: How hard to compress (GPa)
- $V_0$: Most stable volume (Å³)
- $B'_0$: Typically 3-6 for most materials

### Volume Sampling

Typical strategy:
- **Range**: ±6% around initial volume (0.94 to 1.06 scale factors)
- **Points**: 7-11 volumes (more = better fit)
- **Spacing**: Uniform in scale factor

Example scales: [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]

### EOS Fitting

1. Calculate energy at each volume
2. Fit to Birch-Murnaghan (or other EOS)
3. Extract B₀, V₀, B'₀
4. Plot E vs V curve

---

## Tutorial Subdirectories

### [01-SiestaEosFlowMaker](01-SiestaEosFlowMaker/)
**Description**: Basic EOS workflow with automatic plotting
**Features**:
- Volume scaling and relaxation
- Birch-Murnaghan fitting
- Automatic EOS plot generation
- Bulk modulus extraction

### [02-EOSBasisConvergenceFlowMaker](02-EOSBasisConvergenceFlowMakere/)
**Description**: EOS with simultaneous basis set testing
**Features**:
- Test multiple basis sets (SZ, DZ, DZP, TZP)
- EOS for each basis
- Identify basis-dependent bulk properties
- Comprehensive comparison plots

### [03-EOSFullBasisConvergenceFlowMaker](03-EOSFullBasisConvergenceFlowMaker/)
**Description**: Complete basis parameter convergence with EOS
**Features**:
- Full basis parameter space exploration
- Combined convergence + EOS workflow
- Production-ready parameter determination

---

## Quick Start

### Example 1: Basic EOS Calculation

```python
from pymatgen.core import Structure
from atomate2.siesta.flows.eos import SiestaEosFlowMaker
from jobflow import run_locally

# Load structure
structure = Structure.from_file("../../00-structures/Si.cif")

# Create EOS workflow
eos_flow = SiestaEosFlowMaker(
    scales=[0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06],  # 7 volumes
    relax_maker_kwargs={
        "user_params": {
            "PAO.BasisSize": "DZP",
            "kpts": [8, 8, 8],
            "Mesh.Cutoff": "300 Ry",
        }
    },
    dry_run=True
)

# Generate and run
workflow = eos_flow.make(structure)
results = run_locally(workflow, create_folders=True)
```

### Example 2: Custom Volume Range

```python
# Larger volume range for soft materials
eos_flow = SiestaEosFlowMaker(
    scales=[0.90, 0.94, 0.98, 1.00, 1.02, 1.06, 1.10],  # ±10%
    dry_run=True
)

# More points for accurate fitting
eos_flow = SiestaEosFlowMaker(
    scales=[0.94, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.06],  # 11 points
    dry_run=True
)
```

### Example 3: With Tier Preset

```python
from atomate2.siesta.sets.tiers import apply_tier_preset
from atomate2.siesta.jobs.core import RelaxMaker

# Create relax maker with preset
relax_maker = RelaxMaker.fixed_cell_relaxation()
relax_maker = apply_tier_preset(relax_maker, "relax_standard")

# Use in EOS workflow
eos_flow = SiestaEosFlowMaker(
    scales=[0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06],
    relax_maker=relax_maker,
    dry_run=True
)
```

### Example 4: EOS with Basis Convergence

```python
from atomate2.siesta.flows.eos import EOSBasisConvergenceFlowMaker

# Test EOS with multiple basis sets
eos_basis_flow = EOSBasisConvergenceFlowMaker(
    basis_sizes=["DZ", "DZP", "TZP"],
    scales=[0.96, 0.98, 1.00, 1.02, 1.04],  # Fewer volumes per basis
    dry_run=True
)

workflow = eos_basis_flow.make(structure)
```

---

## Run Modes

### 1. Dry-Run Mode

```bash
python eos_workflow.py  # With dry_run=True
```

**Output**:
```
preview_output/
├── job_relax_scale_0.94_*/
├── job_relax_scale_0.96_*/
├── job_relax_scale_0.98_*/
├── job_relax_scale_1.00_*/
├── job_relax_scale_1.02_*/
├── job_relax_scale_1.04_*/
├── job_relax_scale_1.06_*/
└── job_eos_analysis_*/      # Analysis job (plots after calculation)
```

**Verify volumes**:
```bash
for dir in preview_output/job_relax_*/; do
    echo "$(basename $dir):"
    grep "LatticeConstant" $dir/siesta.fdf
done
```

### 2. Local Execution

```bash
# Edit: Set dry_run=False
python eos_workflow.py
```

**Time**: 30-60 min for 7 volumes (depends on system size)

---

## Expected Output

### Automatic EOS Plot

After workflow completes, check analysis job folder:

```
job_eos_analysis_*/
├── eos_plot.png             # E vs V curve with fitted EOS
├── eos_summary.txt          # Bulk modulus, V₀, B'₀
└── eos_data.json            # Raw data for further analysis
```

**eos_plot.png** shows:
- Data points (calculated E vs V)
- Fitted Birch-Murnaghan curve
- Equilibrium volume V₀ marked
- Bulk modulus B₀ annotated

### Analyzing EOS Results

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Read EOS results
with open("job_eos_analysis_*/eos_data.json") as f:
    eos_data = json.load(f)

volumes = np.array(eos_data['volumes'])
energies = np.array(eos_data['energies'])
v0 = eos_data['equilibrium_volume']
b0 = eos_data['bulk_modulus_gpa']
b0_prime = eos_data['bulk_modulus_derivative']

print(f"Equilibrium volume: {v0:.3f} Å³")
print(f"Bulk modulus: {b0:.1f} GPa")
print(f"B'₀: {b0_prime:.2f}")

# Convert to lattice parameter for cubic system
a0 = v0**(1/3)
print(f"Equilibrium lattice parameter: {a0:.4f} Å")
```

### Extracting Per-Atom Properties

```python
n_atoms = len(structure)

v0_per_atom = v0 / n_atoms
e0_per_atom = eos_data['equilibrium_energy'] / n_atoms

print(f"Equilibrium volume per atom: {v0_per_atom:.3f} Å³/atom")
print(f"Cohesive energy per atom: {e0_per_atom:.4f} eV/atom")
```

### Comparison with Experiments

```python
# Example: Silicon
exp_a0 = 5.431  # Å (experimental)
calc_a0 = v0**(1/3)

error = (calc_a0 - exp_a0) / exp_a0 * 100
print(f"Calculated a₀: {calc_a0:.4f} Å")
print(f"Experimental a₀: {exp_a0:.4f} Å")
print(f"Error: {error:+.2f}%")
# GGA typically overestimates lattice parameters by ~1-2%
```

---

## Common Issues

### Issue 1: "EOS fitting fails"

**Symptoms**: Poor fit, unrealistic B₀ or V₀

**Solutions**:

1. **Increase volume range**:
   ```python
   scales=[0.90, 0.94, 0.98, 1.00, 1.02, 1.06, 1.10]  # ±10% instead of ±6%
   ```

2. **Add more points**:
   ```python
   scales=[0.94, 0.96, 0.98, 0.99, 1.00, 1.01, 1.02, 1.04, 1.06]  # 9 instead of 7
   ```

3. **Check for structural instabilities**:
   ```bash
   grep "Relaxed" job_relax_*/siesta.out
   # All should show successful relaxation
   ```

4. **Try different EOS forms** (manual):
   - Birch-Murnaghan (default, best for most materials)
   - Vinet (good for anharmonic materials)
   - Murnaghan (simple, less accurate)

### Issue 2: "Volume at boundary of range"

**Symptoms**: V₀ at or near smallest/largest volume

**Solution**: Expand volume range in that direction
```python
# If V₀ near 0.94, go lower
scales=[0.88, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]
```

### Issue 3: "Energies not monotonic"

**Symptoms**: E(V) curve has unexpected jumps

**Causes**:
- Relaxation to different structures at different volumes
- SCF convergence failures
- Insufficient k-points or basis

**Solutions**:
1. Check all relaxations converged:
   ```bash
   for dir in job_relax_*/; do
       echo "$dir:"
       grep "SCF Converge" $dir/siesta.out | tail -1
   done
   ```

2. Use tighter parameters:
   ```python
   relax_maker_kwargs={
       "user_params": {
           "kpts": [10, 10, 10],      # Denser
           "DM.Tolerance": "1e-6",    # Tighter
       }
   }
   ```

### Issue 4: "Bulk modulus too high/low"

**Cause**: GGA functional, unconverged parameters

**Solutions**:
1. **Accept GGA limitations**: Bulk moduli typically accurate within ~10%
2. **Check convergence**: Run [01-convergence](../01-convergence/) first
3. **Compare trends**: Relative values more reliable than absolute

---

## EOS Analysis Best Practices

### 1. Always Use Relaxation

```python
# ✅ CORRECT - Relax at each volume
eos_flow = SiestaEosFlowMaker(
    scales=[...],
    relax=True  # Default
)

# ❌ WRONG - Static calculations give wrong EOS
eos_flow = SiestaEosFlowMaker(
    scales=[...],
    relax=False  # Don't do this!
)
```

### 2. Use Converged Parameters

From convergence studies ([01-convergence](../01-convergence/)):
```python
relax_maker_kwargs={
    "user_params": {
        "kpts": [10, 10, 10],      # Converged value
        "Mesh.Cutoff": "350 Ry",   # Converged value
        "PAO.BasisSize": "DZP",    # Converged value
    }
}
```

### 3. Check Symmetry Preservation

```python
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Check all volumes maintain symmetry
for scale in scales:
    scaled_structure = structure.scale_lattice(structure.volume * scale)
    sga = SpacegroupAnalyzer(scaled_structure)
    print(f"Scale {scale}: Space group {sga.get_space_group_symbol()}")
```

### 4. Validate with Experiments

```python
# Compare with experimental data
exp_b0 = 98  # GPa for Si
calc_b0 = eos_data['bulk_modulus_gpa']

agreement = abs(calc_b0 - exp_b0) / exp_b0 * 100
print(f"Bulk modulus agreement with experiment: {100-agreement:.1f}%")
```

---

## Tips for Success

✅ **Use converged parameters**: Run convergence tests first
✅ **Adequate volume range**: ±6% minimum, ±8% safer
✅ **Enough points**: 7-9 minimum, 11 better
✅ **Check all relaxations**: Verify convergence at each volume
✅ **Compare with experiments**: Validate B₀ and V₀
✅ **Use tier presets**: Material-specific optimized parameters

---

## Next Steps

After completing EOS tutorials:

1. **Mechanical properties**: [04-mechanical](../04-mechanical/) - Full elastic tensor
2. **Thermal expansion**: Combine EOS with [06-vibrational-properties/03-qha](../06-vibrational-properties/03-SiestaQhaFlowMaker/)
3. **Pressure effects**: Use EOS to understand material under pressure
4. **Recipe book**: [03-advanced-features/08-recipe-book](../../03-advanced-features/08-recipe-book/) - One-line EOS workflows

---

## References

- **Birch-Murnaghan EOS**: Birch (1947) "Finite Elastic Strain of Cubic Crystals"
- **Bulk modulus**: Ashcroft & Mermin "Solid State Physics"
- **EOS fitting**: Vinet et al. (1987) J. Geophys. Res.

---

*Back to [02-workflows](../README.md) | [Main Tutorial Index](../../README.md)*
