# Tutorial: Combined Mesh Cutoff + K-Points Convergence

**Category**: 02-workflows/01-convergence/MeshKpointConvergenceFlowMaker
**Difficulty**: Intermediate
**Time**: ~10 min (dry-run), ~30-60 min (full calculation)

---

## Overview

This tutorial demonstrates combined two-stage convergence testing using `MeshKpointConvergenceFlowMaker`. This workflow first converges the mesh cutoff, then uses the converged cutoff to find optimal k-points - ensuring both parameters are properly optimized together.

**Key Goal**: Systematically determine optimal mesh cutoff AND k-points in one workflow.

---

## What You'll Learn

- Two-stage convergence workflow (mesh cutoff → k-points)
- Intelligent early stopping when convergence is reached
- Multi-property convergence criteria (energy, Fermi, forces, stress, bandgap)
- Automatic convergence analysis with 6 plots per stage
- Production-ready parameter determination

---

## Prerequisites

- **Required**: Understanding of [mesh cutoff](../MeshCutoffConvergenceFlowMaker/) and [k-points](../KpointsConvergenceFlowMaker/) concepts
- **Recommended**: Complete individual convergence tutorials first
- **Structure files**: Si_mp-149_primitive.cif

---

## Key Concepts

### Two-Stage Approach

**Stage 1: Mesh Cutoff Convergence**
- Tests multiple mesh cutoff values
- Uses coarse, fixed k-points (typically 1×1×1 or 2×2×2)
- Finds minimum converged mesh cutoff
- **Early stopping**: Stops when 2 consecutive cutoffs are converged

**Stage 2: K-Points Convergence**
- Uses converged mesh cutoff from Stage 1
- Tests multiple k-point meshes
- Finds optimal k-point density
- **Early stopping**: Stops when 2 consecutive k-meshes are converged

### Why Two Stages?

**Problem**: Testing all combinations is expensive
- 7 mesh cutoffs × 6 k-point meshes = 42 calculations

**Solution**: Sequential optimization
- Stage 1: 7 mesh cutoff tests (with coarse k-points)
- Stage 2: 6 k-point tests (with converged cutoff)
- Total: ~13 calculations (saves many of time!)

### Convergence Criteria

**Default (Energy Only)**:
- Energy difference < 1.0 meV between consecutive tests
- Require 2 consecutive converged points

**Multi-Property** (Optional):
- Energy: < 1.0 meV
- Fermi energy: < 0.01 eV
- Max force: < 0.01 eV/Å
- Max stress: < 0.05 eV/Å³
- Bandgap: < 0.01 eV (if applicable)

---

## Tutorial Files

This directory contains **1 example**:

### `MeshKpointConvergenceFlowMaker.py`

**Description**: Combined two-stage convergence with energy-only criteria

**Stage 1 Parameters**:
- Mesh cutoffs: [200, 250, 300, 350, 400] Ry
- K-points (fixed): [1, 1, 1] (very coarse for speed)

**Stage 2 Parameters**:
- K-points: [[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8]]
- Mesh cutoff: Converged value from Stage 1

**Convergence**: ΔE < 1.0 meV, require 2 consecutive

**Time**: ~30-60 minutes for full workflow

---

## Quick Start

### Basic Example (Energy-Only Convergence)

```python
from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.convergence import MeshKpointConvergenceFlowMaker

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

maker = MeshKpointConvergenceFlowMaker(
    mesh_cutoffs=[200, 250, 300, 350, 400],
    kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8]],
    stage1_kpoints=[1, 1, 1],  # Coarse k-points for mesh convergence
    dry_run=False,
)

flow = maker.make(structure)
results = run_locally(flow, create_folders=True)
```

### With Multi-Property Convergence

```python
from atomate2.siesta.flows.convergence import (
    MeshKpointConvergenceFlowMaker,
    ConvergenceCriteria,
)

# Define strict convergence for all properties
criteria = ConvergenceCriteria(
    energy_tol=1.0,     # 1 meV energy
    fermi_tol=0.01,     # 0.01 eV Fermi energy
    force_tol=0.01,     # 0.01 eV/Å maximum force
    stress_tol=0.05,    # 0.05 eV/Å³ maximum stress
)

maker = MeshKpointConvergenceFlowMaker(
    mesh_cutoffs=[200, 250, 300, 350, 400],
    kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10]],
    stage1_kpoints=[2, 2, 2],
    convergence_criteria=criteria,
    require_consecutive=2,  # Need 2 consecutive converged points
)

flow = maker.make(structure)
results = run_locally(flow, create_folders=True)
```

---

## Expected Output

### Automatic Plots (12 Total: 6 per Stage)

**Stage 1: Mesh Cutoff Convergence**
1. `convergence_mesh_cutoff_energy.png` - Energy vs. mesh cutoff
2. `convergence_mesh_cutoff_convergence.png` - Convergence check (meV from finest)
3. `convergence_mesh_cutoff_fermi.png` - Fermi energy tracking
4. `convergence_mesh_cutoff_bandgap.png` - Band gap evolution (if applicable)
5. `convergence_mesh_cutoff_forces.png` - Maximum force tracking
6. `convergence_mesh_cutoff_stress.png` - Maximum stress tracking

**Stage 2: K-Points Convergence**
1. `convergence_kpoints_energy.png` - Energy vs. k-points
2. `convergence_kpoints_convergence.png` - Convergence check
3. `convergence_kpoints_fermi.png` - Fermi energy tracking
4. `convergence_kpoints_bandgap.png` - Band gap evolution
5. `convergence_kpoints_forces.png` - Maximum force tracking
6. `convergence_kpoints_stress.png` - Maximum stress tracking

### Summary Files

**Stage 1** (`convergence_mesh_cutoff.txt`):
```
Stage 1: Mesh Cutoff Convergence
=================================

Tested mesh cutoffs: 5
  200 Ry  →  -10.232456 eV/atom
  250 Ry  →  -10.238123 eV/atom
  300 Ry  →  -10.238891 eV/atom (CONVERGED)
  350 Ry  →  -10.238956 eV/atom (CONVERGED)
  400 Ry  →  [SKIPPED - Early stopping triggered]

Convergence threshold: 1.0 meV/atom
First converged cutoff: 300 Ry
Consecutive converged: 2 (300, 350 Ry)

→ Using 350 Ry for Stage 2 (k-points convergence)
```

**Stage 2** (`convergence_kpoints.txt`):
```
Stage 2: K-Points Convergence
==============================

Using converged mesh cutoff: 350 Ry

Tested k-point meshes: 4
  2×2×2  →  -10.234567 eV/atom
  4×4×4  →  -10.237891 eV/atom
  6×6×6  →  -10.238901 eV/atom (CONVERGED)
  8×8×8  →  -10.238975 eV/atom (CONVERGED)

Convergence threshold: 1.0 meV/atom
First converged mesh: 6×6×6
Consecutive converged: 2 (6×6×6, 8×8×8)

FINAL RECOMMENDATION:
  Mesh.Cutoff = 350 Ry
  K-points = 6×6×6
```

---

## Early Stopping Feature

**How it works**:
1. Tests proceed in order (increasing cutoff/k-density)
2. After each test, checks if last N consecutive points are converged
3. If yes, stops testing and uses last converged value
4. Saves computational time by avoiding unnecessary tests

**Example**:
```
Mesh cutoffs planned: [200, 250, 300, 350, 400, 450, 500]
Convergence requirement: 2 consecutive within 1 meV

Results:
  200 Ry:  -10.232 eV (reference)
  250 Ry:  -10.238 eV (6 meV change - not converged)
  300 Ry:  -10.239 eV (1 meV change - converged!)
  350 Ry:  -10.239 eV (0.2 meV change - converged!)

  → 2 consecutive converged! Stop here.
  → Skip 400, 450, 500 Ry tests
  → Use 350 Ry for Stage 2

Time saved: 3 calculations × ~5 min = ~15 minutes!
```

---

## Common Configurations

### Fast Test (Insulators/Semiconductors)
```python
maker = MeshKpointConvergenceFlowMaker(
    mesh_cutoffs=[200, 250, 300, 350],  # Fewer points
    kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6]],
    stage1_kpoints=[1, 1, 1],
    require_consecutive=2,
)
```

### Standard Test (General Purpose)
```python
maker = MeshKpointConvergenceFlowMaker(
    mesh_cutoffs=[200, 250, 300, 350, 400, 450],
    kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8]],
    stage1_kpoints=[2, 2, 2],
    require_consecutive=2,
)
```

### Thorough Test (Metals/Difficult Systems)
```python
maker = MeshKpointConvergenceFlowMaker(
    mesh_cutoffs=[250, 300, 350, 400, 450, 500],
    kpoints_list=[[4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10], [12, 12, 12]],
    stage1_kpoints=[4, 4, 4],  # Denser for metals
    require_consecutive=3,  # More stringent
)
```

---

## Running the Example

```bash
cd MeshKpointConvergenceFlowMaker
python MeshKpointConvergenceFlowMaker.py
```

**What happens**:
1. **Stage 1**: Tests mesh cutoffs with coarse k-points
2. **Early stopping**: Stops when 2 consecutive cutoffs converge
3. **Stage 2**: Tests k-points with converged cutoff from Stage 1
4. **Early stopping**: Stops when 2 consecutive k-meshes converge
5. **Analysis**: Generates 12 plots + 2 summary files

**Check results**:
```bash
ls job_*/convergence_*.png
cat job_*/convergence_mesh_cutoff.txt
cat job_*/convergence_kpoints.txt
```

---

## Common Issues

### Issue 1: "Stage 1 Tests All Cutoffs (No Early Stopping)"

**Symptoms**: All planned mesh cutoffs are tested despite convergence

**Cause**: Consecutive convergence requirement not met

**Solutions**:
1. **Relax convergence requirement**:
   ```python
   require_consecutive=1  # Instead of 2
   ```

2. **Looser tolerance**:
   ```python
   convergence_criteria=ConvergenceCriteria(energy_tol=2.0)  # 2 meV instead of 1
   ```

### Issue 2: "Stage 2 Uses Wrong Cutoff"

**Symptoms**: K-point tests don't use converged cutoff from Stage 1

**Cause**: Early stopping didn't trigger, using last value from list

**Solution**: Check Stage 1 summary - if not converged, extend mesh_cutoffs range
```python
mesh_cutoffs=[200, 250, 300, 350, 400, 450, 500]  # Add more values
```

### Issue 3: "Multi-Property Criteria Never Converge"

**Symptoms**: Tests all values, never triggers early stopping

**Cause**: One property (often forces/stress) not converging

**Solutions**:
1. **Check which property fails**:
   - Look at individual plots (forces, stress, etc.)
   - Identify problematic property

2. **Relax that property's tolerance**:
   ```python
   criteria = ConvergenceCriteria(
       energy_tol=1.0,
       fermi_tol=0.01,
       force_tol=0.05,  # Relaxed from 0.01
       stress_tol=0.10,  # Relaxed from 0.05
   )
   ```

3. **Use energy-only for difficult systems**:
   ```python
   criteria = ConvergenceCriteria(energy_tol=1.0)  # Only energy
   ```

### Issue 4: "Different Properties Converge at Different Points"

**Symptoms**: Energy converges at 300 Ry, but forces need 400 Ry

**Cause**: Forces more sensitive to mesh cutoff than energy

**Solution**: This is normal! The workflow uses the **highest** converged value
```
Energy converged at: 300 Ry
Forces converged at: 400 Ry
→ Workflow uses: 400 Ry (ensures ALL properties converged)
```

---

## Best Practices

✅ **Start with energy-only**: Test basic convergence first, then add multi-property if needed

✅ **Use coarse Stage 1 k-points**: 1×1×1 or 2×2×2 sufficient for mesh cutoff testing

✅ **Plan extra test points**: Early stopping may not trigger, better to have extra than too few

✅ **Check both summary files**: Verify both stages converged properly

✅ **Verify plots**: Visual inspection confirms convergence behavior

✅ **Use for production**: These parameters are rigorously tested

❌ **Don't use same k-points in both stages**: Stage 1 should use coarser k-points

❌ **Don't skip individual tests first**: Understand mesh cutoff and k-points separately before combined

❌ **Don't trust partial convergence**: If only Stage 1 converges, need to extend Stage 2 range

---

## Interpreting Results

### Good Convergence Example

**Stage 1**:
```
300 Ry: CONVERGED
350 Ry: CONVERGED ← Early stop triggered, use this
400 Ry: SKIPPED
```

**Stage 2** (using 350 Ry):
```
6×6×6:  CONVERGED
8×8×8:  CONVERGED ← Early stop triggered, use this
10×10×10: SKIPPED
```

**Final recommendation**: 350 Ry, 6×6×6 k-points ✓

### Problematic Convergence Example

**Stage 1**:
```
300 Ry: NOT converged (2.5 meV change)
350 Ry: NOT converged (1.8 meV change)
400 Ry: CONVERGED ← Last value, but only 1 consecutive
```

**Action needed**: Extend mesh_cutoffs to include 450, 500 Ry

---

## Next Steps

After combined convergence:

1. **Basis parameters**: [BasisParametersConvergenceFlowMaker](../BasisParametersConvergenceFlowMaker/) - Use converged mesh cutoff and k-points
2. **Production calculations**: Apply converged parameters to:
   - [EOS workflows](../../02-equation-of-states/)
   - [Elastic constants](../../04-mechanical/)
   - [Phonon calculations](../../06-vibrational-properties/)
3. **Tier presets**: Consider using [tier system](../../../03-advanced-features/01-tier-system/) with your converged values

---

## Advanced: Custom Convergence Criteria

For specialized calculations:

**Phonon calculations** (forces critical):
```python
criteria = ConvergenceCriteria(
    energy_tol=1.0,
    force_tol=0.001,  # Very tight for accurate forces!
    stress_tol=0.01,
)
```

**Band structure** (Fermi/bandgap critical):
```python
criteria = ConvergenceCriteria(
    energy_tol=1.0,
    fermi_tol=0.005,   # Tight Fermi energy
    bandgap_tol=0.005, # Tight bandgap
)
```

**Stress/elastic** (stress critical):
```python
criteria = ConvergenceCriteria(
    energy_tol=1.0,
    stress_tol=0.01,  # Very tight for stress tensor!
)
```

---

## References

1. **Convergence testing**: Lejaeghere et al., *Science* 351, aad3000 (2016)
2. **SIESTA convergence**: Soler et al., *J. Phys.: Condens. Matter* 14, 2745 (2002)

---

## Summary

**Key Takeaways**:
- Two-stage approach saves many of computational time
- Early stopping prevents unnecessary calculations
- Multi-property criteria ensure all properties converged
- Energy-only sufficient for most total energy calculations
- Forces/stress criteria important for phonons/elastic properties

**Workflow advantages**:
- Systematic: Tests both parameters properly
- Efficient: Early stopping saves time
- Flexible: Support multiple convergence criteria
- Production-ready: Generates publication-quality plots and summaries

---

*Back to [Convergence Studies](../README.md) | [Main Tutorial Index](../../../README.md)*
