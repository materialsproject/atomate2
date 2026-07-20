# Tutorial: Mesh Cutoff Convergence

**Category**: 02-workflows/01-convergence/MeshCutoffConvergenceFlowMaker
**Difficulty**: Beginner
**Time**: ~5 min (dry-run), ~10-15 min (full calculation)

---

## Overview

This tutorial demonstrates mesh cutoff convergence testing using `MeshCutoffConvergenceFlowMaker`. The mesh cutoff parameter controls the fineness of the real-space grid used to represent wavefunctions and charge density in SIESTA calculations.

**Key Goal**: Determine optimal mesh cutoff that provides accurate energies without excessive computational cost.

---

## What You'll Learn

- Mesh cutoff convergence testing
- Real-space grid fineness optimization
- Energy convergence criteria (meV/atom)
- Automatic convergence plotting and analysis
- Optimal cutoff selection for different pseudopotentials

---

## Prerequisites

- **Required**: Basic understanding of real-space grids in DFT
- **Recommended**: Complete [RelaxMaker tutorials](../../../01-basics/01-RelaxMaker/) first
- **Structure files**: Si_mp-149_primitive.cif

---

## Key Concepts

### Mesh Cutoff (Mesh.Cutoff)

**Purpose**: Controls fineness of real-space grid for representing wavefunctions and charge density

**Units**: Ry (Rydberg) or eV
- **Conversion**: 1 Ry = 13.6057 eV

**Effect**:
- Higher cutoff = finer grid = more accurate
- Computational cost scales as O(N_grid)
- Typical range: 100-500 Ry (200-300 Ry is common)

**Convergence Criterion**: Energy change < 1-5 meV/atom between successive cutoff values

### Test Range

Typical mesh cutoff test values:
- **Start**: 100 Ry (coarse, fast - just to see trend)
- **Intermediate**: 150, 200, 250, 300 Ry
- **High accuracy**: 350, 400, 450, 500 Ry

**Rule of thumb**: Start low to see full convergence behavior, extend if not converged by 400 Ry

---

## Tutorial Files

This directory contains **1 basic example**:

### `MeshCutoffConvergenceFlowMaker.py`

**Description**: Basic mesh cutoff convergence test for Silicon

**Parameters**:
- Structure: Si_mp-149_primitive.cif (covalent semiconductor)
- Mesh cutoffs tested: [100, 150, 200, 250, 300, 350, 400] Ry
- K-points: Fixed (typically 4×4×4 or pre-converged value)

**Time**: ~10-15 minutes for 7 cutoff values

---

## Quick Start

### Basic Example

```python
from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.convergence import MeshCutoffConvergenceFlowMaker

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

flow = MeshCutoffConvergenceFlowMaker(
    dry_run=False,
    mesh_cutoffs=[100, 150, 200, 250, 300, 350, 400],
)
workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True)
```

### With Fixed K-Points

```python
from atomate2.siesta.powerups import update_user_siesta_settings

flow = MeshCutoffConvergenceFlowMaker(
    mesh_cutoffs=[100, 150, 200, 250, 300, 350, 400],
)
workflow = flow.make(structure)

# Use fixed k-point grid during mesh cutoff testing
workflow = update_user_siesta_settings(
    workflow, {"a2s_kpts": [4, 4, 4]}  # Keep constant!
)

results = run_locally(workflow, create_folders=True)
```

---

## Expected Output

### Automatic Plots

The workflow generates convergence analysis plots:

#### 1. **Energy vs. Mesh Cutoff** (`convergence_mesh_cutoff_energy.png`)
- X-axis: Mesh cutoff (Ry)
- Y-axis: Total energy (eV/atom)
- Shows monotonic decrease toward converged value

#### 2. **Convergence Check** (`convergence_mesh_cutoff_convergence.png`)
- X-axis: Mesh cutoff (Ry)
- Y-axis: Energy difference from finest mesh (meV)
- Horizontal line: Convergence threshold (typically 1 meV)

#### 3. **Fermi Energy** (`convergence_mesh_cutoff_fermi.png`)
- X-axis: Mesh cutoff (Ry)
- Y-axis: Fermi energy (eV)
- Monitors electronic structure consistency

### Summary File (`convergence_mesh_cutoff.txt`)

```
Mesh Cutoff Convergence Summary
================================

Tested mesh cutoffs: 7
  100 Ry  →  -10.185432 eV/atom
  150 Ry  →  -10.228901 eV/atom
  200 Ry  →  -10.236789 eV/atom
  250 Ry  →  -10.238456 eV/atom
  300 Ry  →  -10.238891 eV/atom (CONVERGED)
  350 Ry  →  -10.238967 eV/atom (CONVERGED)
  400 Ry  →  -10.238989 eV/atom (CONVERGED)

Convergence threshold: 1.0 meV/atom
First converged cutoff: 300 Ry

Recommendation: Use 300 Ry for production calculations
(Add 50-100 Ry safety margin: 350-400 Ry for critical work)
```

---

## Common Cutoff Ranges

### Conservative (Fast)
```python
mesh_cutoffs = [150, 200, 250, 300]
```

### Standard (Recommended)
```python
mesh_cutoffs = [100, 150, 200, 250, 300, 350, 400]
```

### High Accuracy
```python
mesh_cutoffs = [200, 250, 300, 350, 400, 450, 500]
```

### Hard Pseudopotentials (May need higher)
```python
mesh_cutoffs = [300, 350, 400, 450, 500, 550, 600]
```

---

## Running the Example

```bash
cd MeshCutoffConvergenceFlowMaker
python MeshCutoffConvergenceFlowMaker.py
```

**What happens**:
- 7 static SCF calculations run in parallel
- Each with different mesh cutoff
- Generates 3 plots + summary file
- Output in `MeshCutoffConvergenceFlowMaker/job_*/` directories

**Check results**:
```bash
ls MeshCutoffConvergenceFlowMaker/job_*/
cat MeshCutoffConvergenceFlowMaker/*_summary.txt
```

---

## Common Issues

### Issue 1: "Energy Doesn't Converge"

**Symptoms**: Energy keeps decreasing even at 400-500 Ry

**Causes**:
- Hard pseudopotentials (transition metals, lanthanides)
- Very localized orbitals
- Need finer mesh

**Solutions**:
1. **Extend cutoff range**:
   ```python
   mesh_cutoffs = [400, 450, 500, 550, 600]
   ```

2. **Check pseudopotential quality**:
   - Some pseudopotentials require higher cutoffs
   - Consider using softer pseudopotentials if available

3. **Verify this level of convergence is needed**:
   - 1 meV/atom for total energies
   - May need tighter for forces/phonons

### Issue 2: "Oscillating Energy"

**Symptoms**: Energy doesn't decrease monotonically

**Causes**:
- Structure not properly relaxed
- K-points causing noise
- Pseudopotential issues

**Solutions**:
1. **Relax structure first**:
   ```python
   from atomate2.siesta.jobs.core import RelaxMaker
   relax = RelaxMaker.variable_cell_relaxation()
   relax_job = relax.make(structure)
   ```

2. **Use denser k-points**:
   ```python
   update_user_siesta_settings(workflow, {"a2s_kpts": [6, 6, 6]})
   ```

3. **Check SCF convergence**:
   - Increase MaxSCFIterations
   - Tighten DM.Tolerance

### Issue 3: "Too Expensive"

**Symptoms**: High cutoff calculations take too long

**Solutions**:
1. **Use coarser basis during testing**:
   ```python
   update_user_siesta_settings(
       workflow, {"PAO.BasisSize": "SZ"}  # Instead of DZP
   )
   ```

2. **Run on HPC cluster**:
   - See [Job Submission tutorials](../../../04-infrastructure/02-job-submission/)

3. **Test subset of cutoffs first**:
   ```python
   mesh_cutoffs = [150, 200, 250, 300, 350]  # Skip 100 and 400
   ```

### Issue 4: "Calculation Crashes at Low Cutoff"

**Symptoms**: 100 Ry or 150 Ry calculations fail

**Cause**: Grid too coarse for basis set

**Solution**: Start from higher cutoff
```python
mesh_cutoffs = [150, 200, 250, 300, 350, 400]  # Skip 100 Ry
```

---

## Best Practices

✅ **Use fixed k-points**: Keep k-point grid constant during mesh cutoff convergence

✅ **Test broad range**: Start from 100 Ry to see full trend (even if it's inaccurate)

✅ **Check monotonic decrease**: Energy should decrease smoothly

✅ **Plot results**: Visual inspection helps identify convergence

✅ **Add safety margin**: Use 50-100 Ry above converged value for production

✅ **Converge first**: Do this before k-point convergence

❌ **Don't use too few points**: At least 5-7 cutoff values recommended

❌ **Don't skip low cutoffs**: Need to see full convergence behavior

❌ **Don't ignore oscillations**: Non-monotonic behavior indicates problems

---

## Interpreting Results

### How to Read Energy vs. Cutoff Plot

**Good convergence**:
- Energy decreases monotonically
- Flattens out at higher cutoffs
- Last 2-3 points differ by < 1 meV

**Example**:
```
100 Ry:  -10.185 eV  (reference, very inaccurate)
150 Ry:  -10.229 eV  (44 meV change - not converged)
200 Ry:  -10.237 eV  (8 meV change - getting close)
250 Ry:  -10.238 eV  (1.7 meV change - nearly converged)
300 Ry:  -10.239 eV  (0.4 meV change - CONVERGED)
350 Ry:  -10.239 eV  (0.08 meV change - converged)
400 Ry:  -10.239 eV  (0.02 meV change - converged)

→ Use 300 Ry for production (or 350 Ry with safety margin)
```

**Poor convergence**:
- Energy still changing significantly at 400 Ry
- Need to extend test range
- May indicate pseudopotential issues

---

## Material-Specific Guidelines

### Soft Elements (C, Si, O, N)
- **Typical**: 200-300 Ry
- **Start from**: 100 Ry
- **Rarely need**: > 400 Ry

### Transition Metals (Fe, Cu, Ni, Ti)
- **Typical**: 300-400 Ry
- **Start from**: 150 Ry
- **May need**: 450-500 Ry for d-orbitals

### Hard Pseudopotentials
- **Typical**: 400-500 Ry
- **Start from**: 200 Ry
- **May need**: 550-600 Ry

### Lanthanides/Actinides
- **Typical**: 400-600 Ry
- **Start from**: 250 Ry
- **May need**: > 600 Ry for f-orbitals

---

## Next Steps

After mesh cutoff convergence:

1. **K-points convergence**: [KpointsConvergenceFlowMaker](../KpointsConvergenceFlowMaker/) - Use converged cutoff
2. **Combined verification**: [MeshKpointConvergenceFlowMaker](../MeshKpointConvergenceFlowMaker/) - Test both together
3. **Basis parameters**: [BasisParametersConvergenceFlowMaker](../BasisParametersConvergenceFlowMaker/) - Fine-tune basis quality
4. **Production calculations**: Use converged cutoff in workflows

---

## Advanced: Pseudopotential-Specific Cutoffs

Different pseudopotentials may require different cutoffs:

```python
# Check recommended cutoff in pseudopotential file
# Look for "Suggested minimum mesh cutoff" in .psml or .psf files

# Example: ONCVPSP pseudopotentials (soft)
mesh_cutoffs = [150, 200, 250, 300]  # Usually converge by 250-300 Ry

# Example: Norm-conserving with semicore (harder)
mesh_cutoffs = [250, 300, 350, 400, 450]  # May need 400-450 Ry
```

---

## References

1. **Real-space grids**: Soler et al., *J. Phys.: Condens. Matter* 14, 2745 (2002)
2. **SIESTA Manual**: Section on Mesh.Cutoff parameter
3. **Pseudopotential theory**: Hamann et al., *Phys. Rev. Lett.* 43, 1494 (1979)

---

## Summary

**Key Takeaways**:
- Mesh cutoff convergence is **first** convergence test (before k-points)
- Energy should decrease monotonically with increasing cutoff
- 200-300 Ry typical for soft elements
- 300-400 Ry typical for transition metals
- Add 50-100 Ry safety margin for production
- Visual inspection of convergence plot is critical

---

*Back to [Convergence Studies](../README.md) | [Main Tutorial Index](../../../README.md)*
