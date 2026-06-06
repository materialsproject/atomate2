# Tutorial: K-Points Convergence

**Category**: 02-workflows/01-convergence/KpointsConvergenceFlowMaker
**Difficulty**: Beginner
**Time**: ~5 min (dry-run), ~10-20 min (full calculation)

---

## Overview

This tutorial demonstrates k-point mesh convergence testing using `KpointsConvergenceFlowMaker`. K-points sample the Brillouin zone for electronic structure calculations, and proper convergence is essential for accurate total energies and band structures.

**Key Goal**: Determine optimal k-point mesh density that balances accuracy and computational cost.

---

## What You'll Learn

- K-point mesh convergence testing
- Brillouin zone sampling optimization
- Energy convergence criteria (meV/atom)
- Automatic convergence plotting and analysis
- Optimal k-mesh selection for different materials

---

## Prerequisites

- **Required**: Basic understanding of k-point sampling in DFT
- **Recommended**: [Mesh Cutoff Convergence](../MeshCutoffConvergenceFlowMaker/) completed first
- **Structure files**: Si_mp-149_primitive.cif

---

## Key Concepts

### K-Points (kpts)

**Purpose**: Sample the Brillouin zone for integration over k-space

**Specification**: `[nx, ny, nz]` - divisions along reciprocal lattice vectors

**Examples**:
- `[2, 2, 2]` - 8 k-points (very coarse)
- `[4, 4, 4]` - 64 k-points (moderate)
- `[8, 8, 8]` - 512 k-points (dense)

**Effect**:
- Denser mesh = better integration = more accurate
- Cost scales linearly with total number of k-points
- Metals require denser sampling than insulators/semiconductors

**Convergence Criterion**: Energy change < 1-5 meV/atom between successive k-meshes

### Material Dependence

**Metals**:
- Require dense k-meshes (6×6×6 to 12×12×12 or higher)
- Fermi surface sampling is critical
- Use Methfessel-Paxton or other smearing methods

**Insulators/Semiconductors**:
- Sparser meshes acceptable (4×4×4 to 8×8×8)
- Band gap helps with convergence
- Less sensitive to k-point density

**2D Materials**:
- Dense in-plane, sparse out-of-plane (e.g., 8×8×1)
- Vacuum spacing prevents interaction

---

## Tutorial Files

This directory contains **1 basic example**:

### `KpointsConvergenceFlowMaker.py`

**Description**: Basic k-point convergence test for Silicon

**Parameters**:
- Structure: Si_mp-149_primitive.cif (covalent semiconductor)
- K-points tested: [2×2×2, 4×4×4, 6×6×6, 8×8×8, 10×10×10]
- Mesh cutoff: Fixed (use converged value, typically 300 Ry)

**Time**: ~10-20 minutes for 5 k-meshes

---

## Quick Start

### Basic Example

```python
from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.convergence import KpointsConvergenceFlowMaker

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

flow = KpointsConvergenceFlowMaker(
    dry_run=False,
    kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10]],
)
workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True)
```

### With Converged Mesh Cutoff

```python
from atomate2.siesta.powerups import update_user_siesta_settings

flow = KpointsConvergenceFlowMaker(
    kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8]],
)
workflow = flow.make(structure)

# Use converged mesh cutoff from previous convergence study
workflow = update_user_siesta_settings(
    workflow, {"Mesh.Cutoff": "300 Ry"}  # Use your converged value!
)

results = run_locally(workflow, create_folders=True)
```

---

## Expected Output

### Automatic Plots

The workflow generates convergence analysis plots:

#### 1. **Energy vs. K-points** (`convergence_kpoints_energy.png`)
- X-axis: Number of k-points or k-mesh density
- Y-axis: Total energy (eV/atom)
- Shows energy convergence trend

#### 2. **Convergence Check** (`convergence_kpoints_convergence.png`)
- X-axis: K-mesh index
- Y-axis: Energy difference from finest mesh (meV)
- Horizontal line: Convergence threshold (typically 1 meV)

#### 3. **Fermi Energy** (`convergence_kpoints_fermi.png`)
- X-axis: K-mesh density
- Y-axis: Fermi energy (eV)
- Monitors electronic structure consistency

### Summary File (`convergence_kpoints.txt`)

```
K-Points Convergence Summary
============================

Tested k-meshes: 5
  2×2×2  →  -10.234567 eV/atom
  4×4×4  →  -10.237891 eV/atom
  6×6×6  →  -10.238901 eV/atom (CONVERGED)
  8×8×8  →  -10.238975 eV/atom (CONVERGED)
  10×10×10 → -10.238989 eV/atom (CONVERGED)

Convergence threshold: 1.0 meV/atom
First converged mesh: 6×6×6

Recommendation: Use 6×6×6 k-points for production calculations
```

---

## Common K-Point Ranges

### Fast Convergence Test (Insulators)
```python
kpoints_list = [[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8]]
```

### Standard Test (Semiconductors)
```python
kpoints_list = [[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [8, 8, 8], [10, 10, 10]]
```

### Dense Test (Metals)
```python
kpoints_list = [[4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10], [12, 12, 12], [14, 14, 14], [16, 16, 16]]
```

### Anisotropic K-Meshes (Layered Materials)
```python
kpoints_list = [
    [2, 2, 4],   # Sparser in z-direction
    [4, 4, 8],
    [6, 6, 12],
    [8, 8, 16],
]
```

---

## Running the Example

```bash
cd KpointsConvergenceFlowMaker
python KpointsConvergenceFlowMaker.py
```

**What happens**:
- 5 static SCF calculations run in parallel
- Each with different k-point mesh
- Generates 3 plots + summary file
- Output in `KpointsConvergenceFlowMaker/job_*/` directories

**Check results**:
```bash
ls KpointsConvergenceFlowMaker/job_*/
cat KpointsConvergenceFlowMaker/*_summary.txt
```

---

## Common Issues

### Issue 1: "Very Slow Convergence"

**Symptoms**: Energy keeps changing even at high k-point densities

**Cause**: Material may be metallic or have complex Fermi surface

**Solutions**:
1. **Increase k-mesh density**:
   ```python
   kpoints_list = [[6, 6, 6], [8, 8, 8], [10, 10, 10], [12, 12, 12], [16, 16, 16]]
   ```

2. **Add smearing for metals**:
   ```python
   workflow = update_user_siesta_settings(
       workflow, {
           "OccupationFunction": "MP",  # Methfessel-Paxton
           "OccupationMPOrder": 1,
           "ElectronicTemperature": "300 K",
       }
   )
   ```

### Issue 2: "Oscillating Energy"

**Symptoms**: Energy doesn't decrease monotonically

**Causes**:
- Structure not well-relaxed
- Mesh cutoff not converged
- Symmetry issues

**Solutions**:
1. **Relax structure first**:
   ```python
   from atomate2.siesta.jobs.core import RelaxMaker
   relax = RelaxMaker.variable_cell_relaxation()
   relax_job = relax.make(structure)
   # Then use relaxed structure for convergence
   ```

2. **Check mesh cutoff convergence**:
   - Run [MeshCutoffConvergenceFlowMaker](../MeshCutoffConvergenceFlowMaker/) first
   - Use converged mesh cutoff value

### Issue 3: "Different Behavior in Different Directions"

**Symptoms**: Need different k-point densities in x, y, z

**Cause**: Anisotropic system (layered materials, 1D chains)

**Solution**: Use anisotropic k-meshes
```python
kpoints_list = [
    [4, 4, 1],   # 2D material (no k-points in z)
    [6, 6, 1],
    [8, 8, 1],
    [12, 12, 1],
]
```

### Issue 4: "Unconverged Even at 16×16×16"

**Symptoms**: Need extremely dense k-meshes

**Causes**:
- Complex metallic Fermi surface
- Topological material
- Need better smearing method

**Solutions**:
1. **Try different smearing**:
   ```python
   # Fermi-Dirac (smooth but needs temperature)
   {"OccupationFunction": "FD", "ElectronicTemperature": "300 K"}

   # Methfessel-Paxton (better for metals)
   {"OccupationFunction": "MP", "OccupationMPOrder": 2}
   ```

2. **Check if this level of convergence is necessary**:
   - For forces/phonons: May need denser k-meshes
   - For total energy: Often 12×12×12 is sufficient

---

## Best Practices

✅ **Use converged mesh cutoff**: Always converge mesh cutoff before k-point testing

✅ **Test sufficient range**: At least 5-7 different k-meshes to see full trend

✅ **Account for symmetry**: Higher symmetry = faster convergence

✅ **Check metallic character**: Metals need denser k-meshes

✅ **Plot results**: Visual inspection is critical for identifying convergence

✅ **Start from coarse**: Include 2×2×2 or 4×4×4 to see full convergence trend

❌ **Don't use unconverged cutoff**: K-point tests meaningless without converged mesh cutoff

❌ **Don't skip coarse meshes**: Need to see full convergence behavior

❌ **Don't over-converge**: Diminishing returns beyond certain density

❌ **Don't forget smearing**: Use appropriate occupation function for metals

---

## Interpreting Results

### How to Read Energy vs. K-Points Plot

**Good convergence**:
- Energy decreases monotonically
- Flattens out at higher k-point densities
- Last 2-3 points differ by < 1 meV

**Example**:
```
2×2×2:   -10.200 eV  (reference)
4×4×4:   -10.235 eV  (35 meV change - not converged)
6×6×6:   -10.238 eV  (3 meV change - nearly converged)
8×8×8:   -10.239 eV  (1 meV change - CONVERGED)
10×10×10: -10.239 eV  (0.2 meV change - converged)

→ Use 8×8×8 for production
```

**Poor convergence**:
- Energy still changing significantly at highest k-mesh
- Oscillations or non-monotonic behavior
- Need to extend test range

---

## Next Steps

After k-point convergence:

1. **Combined convergence**: [MeshKpointConvergenceFlowMaker](../MeshKpointConvergenceFlowMaker/) - Verify both parameters together
2. **Basis parameters**: [BasisParametersConvergenceFlowMaker](../BasisParametersConvergenceFlowMaker/) - Fine-tune basis quality
3. **Production calculations**: Use converged k-points in [EOS](../../02-equation-of-states/), [Elastic](../../04-mechanical/), [Phonons](../../06-vibrational-properties/)

---

## Advanced: Non-Uniform K-Meshes

For special cases (surfaces, interfaces):

```python
# Surface calculation - dense in xy, sparse in z
kpoints_list = [
    [6, 6, 1],
    [8, 8, 1],
    [10, 10, 1],
    [12, 12, 1],
]

# 1D chain - dense along chain, sparse perpendicular
kpoints_list = [
    [1, 1, 8],
    [1, 1, 12],
    [1, 1, 16],
    [1, 1, 20],
]
```

---

## References

1. **Monkhorst-Pack**: Monkhorst & Pack, *Phys. Rev. B* 13, 5188 (1976)
2. **K-point sampling theory**: Chadi & Cohen, *Phys. Rev. B* 8, 5747 (1973)
3. **SIESTA Manual**: Section on k-point sampling

---

## Summary

**Key Takeaways**:
- K-point convergence essential before production calculations
- Metals need denser meshes than insulators
- Always use converged mesh cutoff
- 6×6×6 to 8×8×8 typically sufficient for semiconductors
- Visual inspection of convergence plot is critical

---

*Back to [Convergence Studies](../README.md) | [Main Tutorial Index](../../../README.md)*
