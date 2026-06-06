# Tutorial: Basis Parameters Convergence

**Category**: 02-workflows/01-convergence/BasisParametersConvergenceFlowMaker
**Difficulty**: Intermediate
**Time**: ~15-20 min (simple grid), ~1-2 hours (full grid)

---

## Overview

This tutorial demonstrates systematic optimization of SIESTA basis set parameters (`PAO.EnergyShift` and `PAO.SplitNorm`) using `BasisParametersConvergenceFlowMaker`. These parameters control how SIESTA generates numerical atomic orbitals (PAOs) and significantly affect both accuracy and computational cost.

**Key Goal**: Find optimal parameter combination that minimizes energy while maintaining reasonable basis size.

---

## What You'll Learn

- PAO.EnergyShift: Orbital confinement energy optimization
- PAO.SplitNorm: Basis splitting threshold tuning
- 2D parameter grid convergence testing
- Accuracy vs. basis size trade-offs
- Automatic convergence plots and analysis
- When custodian is essential (difficult systems)

---

## Prerequisites

- **Required**: [K-points and Mesh Cutoff Convergence](../../README.md#1-k-points-convergence) completed
- **Recommended**: Basic understanding of atomic orbital basis sets
- **Structure files**: Si_mp-149_primitive.cif, MgO_mp-1265_primitive.cif

---

## Key Concepts

### PAO.EnergyShift

**Purpose**: Controls orbital confinement energy (how tightly orbitals are confined)

**Units**: Ry (Rydberg) or meV
- 1 Ry = 13.6057 eV = 13605.7 meV

**Effect**:
- **Lower** (0.005 Ry): Larger, more extended orbitals → more accurate → larger basis
- **Higher** (0.02 Ry): Smaller, more confined orbitals → less accurate → smaller basis

**Typical values**:
- Standard: 0.01 Ry (136 meV)
- Tight: 0.005 Ry (68 meV)
- Loose: 0.02 Ry (272 meV)

### PAO.SplitNorm

**Purpose**: Controls basis splitting (single-ζ → double-ζ, polarization orbitals)

**Units**: Dimensionless (norm threshold)

**Effect**:
- **Lower** (0.10): More split orbitals → larger basis → better flexibility
- **Higher** (0.30): Fewer split orbitals → smaller basis → reduced flexibility

**Typical values**:
- Standard: 0.15
- Range: 0.10-0.30

### The Trade-off

```
Lower EnergyShift + Lower SplitNorm
   ↓
More accurate, larger basis, slower calculation

Higher EnergyShift + Higher SplitNorm
   ↓
Less accurate, smaller basis, faster calculation
```

**Goal**: Find the "sweet spot" where energy is converged but basis size is manageable.

---

## Tutorial Files

This directory contains **2 examples** demonstrating different scenarios:

### 1. `BasisParametersConvergenceFlowMaker_01_full_grid.py`

**Description**: Large 5×5 parameter grid for Silicon (easy, covalent system)

**Parameters**:
- Structure: Si_mp-149_primitive.cif (covalent semiconductor)
- Energy shifts: [0.005, 0.01, 0.015, 0.02, 0.025] Ry (5 values)
- Split norms: [0.10, 0.125, 0.15, 0.175, 0.20] (5 values)
- **Total jobs**: 25 SCF calculations
- **Custodian**: Disabled (Si converges easily)

**Why this works without custodian**: Silicon is a well-behaved covalent semiconductor with easy SCF convergence.

**Time**: ~1-2 hours for full 5×5 grid

### 2. `BasisParametersConvergenceFlowMaker_02_with_custodian.py`

**Description**: Smaller 3×3 grid for MgO (hard, ionic system) with automatic error recovery

**Parameters**:
- Structure: MgO_mp-1265_primitive.cif (ionic oxide)
- Energy shifts: [0.01, 0.02, 0.03] Ry (3 values)
- Split norms: [0.10, 0.15, 0.20] (3 values)
- **Total jobs**: 9 SCF calculations
- **Custodian**: **Enabled** (essential for MgO!)
- **Max errors**: 10 recovery attempts per job

**Why custodian is required**: MgO is an ionic oxide with difficult SCF convergence, especially with coarse parameters (2×2×2 k-points, 150 Ry mesh cutoff). Without custodian, calculations will fail.

**Time**: ~30-45 minutes for 3×3 grid

---

## Quick Start

### Example 1: Simple Grid (Silicon, No Custodian)

```python
from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.basis import BasisParametersConvergenceFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

flow = BasisParametersConvergenceFlowMaker(
    energy_shifts=[0.005, 0.01, 0.015, 0.02, 0.025],
    split_norms=[0.10, 0.125, 0.15, 0.175, 0.20],
)
workflow = flow.make(structure)

# Apply coarse parameters for testing
workflow = update_user_siesta_settings(
    workflow, {"a2s_kpts": [2, 2, 2], "Mesh.Cutoff": "150 Ry"}
)

results = run_locally(workflow, create_folders=True, ensure_success=True)
```

### Example 2: With Custodian (MgO, Automatic Error Recovery)

```python
from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.basis import BasisParametersConvergenceFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings

structure = Structure.from_file("../../../00-structures/MgO_mp-1265_primitive.cif")

flow = BasisParametersConvergenceFlowMaker(
    energy_shifts=[0.01, 0.02, 0.03],
    split_norms=[0.10, 0.15, 0.20],
    use_custodian=True,        # Enable automatic error recovery
    custodian_max_errors=10,   # Allow up to 10 recovery attempts
).make(structure)

# Apply coarse parameters
workflow = update_user_siesta_settings(
    flow, {"a2s_kpts": [2, 2, 2], "Mesh.Cutoff": "150 Ry"}
)

results = run_locally(workflow, create_folders=True)
```

---

## Expected Output

### Automatic Plots (4 Visualizations)

The workflow generates comprehensive convergence analysis:

#### 1. **Energy vs. Parameters** (`basis_params_convergence.png`)
Four-panel plot showing:
- **Top-left**: Energy vs. EnergyShift (for each SplitNorm value)
- **Top-right**: Energy vs. SplitNorm (for each EnergyShift value)
- **Bottom-left**: 2D energy landscape heatmap (color shows energy)
- **Bottom-right**: Basis quality map (marker size = basis size, color = forces)

#### 2. **Basis Functions Visualization** (`basis_functions_visualization.png`)
Three-panel plot showing:
- Orbital extent (how far orbitals extend)
- Split-valence distribution (ζ-level distribution)
- Bonding overlap (orbital overlap between atoms)

#### 3. **Real Basis Functions** (from ion.xml files)
- Actual radial wavefunctions for each parameter combination
- Helps visualize how orbitals change with parameters

### Summary File (`basis_params_summary.txt`)

```
Basis Parameters Convergence Summary
====================================

Total parameter combinations tested: 25

Energy Statistics:
  Minimum energy: -214.523456 eV (ES=0.010, SN=0.15)
  Maximum energy: -214.511234 eV (ES=0.025, SN=0.25)
  Energy range: 12.2 meV

Optimal parameters:
  PAO.EnergyShift = 0.010000 Ry (136.1 meV)
  PAO.SplitNorm = 0.1500
  Total energy = -214.523456 eV

Recommendation: Use ES=0.01 Ry, SN=0.15 for production calculations
```

### Timing Information

When custodian is enabled, timing is extracted from `siesta_compressed/siesta.times.gz`. When disabled, timing comes from `siesta.out`.

**Note**: Timing extraction works correctly in both modes.

---

## Common Parameters for Different Materials

### Covalent Systems (Si, C, Diamond)

```python
energy_shifts = [0.010, 0.012, 0.015, 0.018, 0.020]
split_norms = [0.12, 0.15, 0.18, 0.20, 0.22]
```

**Expected optimal**: ES ≈ 0.01-0.015 Ry, SN ≈ 0.15

### Ionic Systems (MgO, NaCl, ZnO)

```python
energy_shifts = [0.005, 0.008, 0.010, 0.012, 0.015]
split_norms = [0.15, 0.18, 0.20, 0.22, 0.25]
```

**Expected optimal**: ES ≈ 0.005-0.01 Ry, SN ≈ 0.15-0.20
**Why tighter**: Ionic bonds require more localized, flexible basis

### Metallic Systems (Al, Cu, Au)

```python
energy_shifts = [0.012, 0.015, 0.018, 0.020, 0.022]
split_norms = [0.15, 0.18, 0.20, 0.22, 0.25]
```

**Expected optimal**: ES ≈ 0.015-0.02 Ry, SN ≈ 0.15
**Why looser**: Delocalized electrons, less sensitive to basis quality

---

## Running the Examples

### 1. Without Custodian (Silicon)

```bash
cd BasisParametersConvergenceFlowMaker
python BasisParametersConvergenceFlowMaker_01_full_grid.py
```

**What happens**:
- 25 SCF calculations run in parallel
- All should converge successfully (Si is easy)
- Generates 4 plots + summary file
- Output in `01/job_*/` directories

**Check results**:
```bash
ls 01/job_*/
cat 01/job_*/basis_params_summary.txt
```

### 2. With Custodian (MgO)

```bash
python BasisParametersConvergenceFlowMaker_02_with_custodian.py
```

**What happens**:
- 9 SCF calculations run in parallel
- Some may initially fail (ionic system is hard)
- **Custodian automatically recovers** via:
  - Removing old DM files
  - Adjusting SCF mixing parameters
  - Increasing MaxSCFIterations
  - Changing mixer weight
- All calculations succeed after recovery
- Output files compressed in `siesta_compressed/`

**Check custodian activity**:
```bash
ls job_*/siesta_compressed/custodian.json.gz
gunzip -c job_*/siesta_compressed/custodian.json.gz | grep -i handler
```

You'll see: `"SCFConvergenceHandler"` if recovery was triggered!

---

## Common Issues

### Issue 1: "SCF Not Converged" (Without Custodian)

**Symptoms**: Job fails with "SCF cycle did not converge"

**Cause**: Parameters too tight or system too difficult

**Solutions**:
1. **Enable custodian** (recommended):
   ```python
   flow = BasisParametersConvergenceFlowMaker(
       ...,
       use_custodian=True,
       custodian_max_errors=10,
   )
   ```

2. **Use easier parameters**:
   ```python
   energy_shifts = [0.015, 0.020, 0.025]  # Looser (smaller basis)
   split_norms = [0.20, 0.25, 0.30]       # Looser (less splitting)
   ```

3. **Improve k-points/cutoff**:
   ```python
   update_user_siesta_settings(
       workflow, {"a2s_kpts": [4, 4, 4], "Mesh.Cutoff": "200 Ry"}
   )
   ```

### Issue 2: "Could Not Find Timing" (With Custodian)

**Symptoms**: Log shows `"Could not find timing for ... Available parameter mappings: []"`

**Cause**: Timing in compressed file with different format

**Solution**: The code checks:
- `siesta.out` (uncompressed)
- `siesta.out.gz` (compressed)
- `siesta_compressed/siesta.times.gz` (custodian compressed)
- Recognizes both `"Elapsed wall time"` and `"Total elapsed wall-clock time"` formats

If you still see this error, update your atomate2siesta installation!

### Issue 3: "Energy Landscape is Noisy"

**Symptoms**: 2D heatmap shows irregular energy surface

**Causes**:
1. K-points not converged
2. Mesh cutoff not converged
3. SCF convergence issues

**Solutions**:
1. **Run k-points/cutoff convergence first**:
   - See [Main Convergence README](../README.md#detailed-tutorial-guides)
   - Get converged k-points and cutoff values
   - Use those in your basis parameter tests

2. **Tighten SCF tolerance**:
   ```python
   update_user_siesta_settings(
       workflow, {
           "DM.Tolerance": 1e-5,      # Tighter (default: 1e-4)
           "MaxSCFIterations": 300,    # More iterations
       }
   )
   ```

### Issue 4: "Basis Too Large" (> 1000 Orbitals)

**Symptoms**: SIESTA reports huge number of orbitals, very slow

**Cause**: EnergyShift too low or SplitNorm too low

**Solutions**:
- Increase EnergyShift (tighter confinement)
- Increase SplitNorm (less splitting)
- Check if such large basis is actually necessary

### Issue 5: "All Parameters Give Same Energy"

**Symptoms**: Energy barely changes across parameter grid

**Causes**:
1. System already converged with DZP basis
2. k-points/cutoff are the limiting factor

**Solutions**:
- This is actually good news! Use standard parameters (ES=0.01, SN=0.15)
- Focus on converging k-points and cutoff instead

---

## Interpreting Results

### How to Read the 2D Heatmap

**X-axis**: PAO.EnergyShift (Ry)
**Y-axis**: PAO.SplitNorm
**Color**: Total energy (darker = lower = better)

**What to look for**:
1. **Minimum energy point**: Darkest spot (optimal parameters)
2. **Flat region around minimum**: Good! Shows parameters are stable
3. **Sharp gradients**: Bad! Energy very sensitive to small changes
4. **Monotonic decrease**: Need tighter parameters (extend grid)

### Example Interpretation

```
Good result:
  - Minimum at ES=0.01, SN=0.15
  - Energy flat within ±0.002 Ry of minimum
  - Nearby points show similar energies
  → Use ES=0.01, SN=0.15 for production

Bad result:
  - Energy keeps decreasing toward ES=0.005
  - No clear minimum in tested range
  - Large energy differences between nearby points
  → Need to test tighter parameters (smaller ES values)
```

### Basis Size Considerations

Check SIESTA output:
```bash
grep "Number of atomic orbitals" 01/job_*/siesta.out
```

**Typical values**:
- DZ: ~100-200 orbitals (for small systems)
- DZP: ~200-400 orbitals
- TZP: ~400-800 orbitals

**With parameter variations**:
- Lower EnergyShift: +20-50% more orbitals
- Lower SplitNorm: +10-30% more orbitals

**Acceptable**: 100-500 orbitals for most systems
**Warning**: > 1000 orbitals may indicate over-optimization

---

## Best Practices

✅ **Converge k-points and cutoff first**: Don't test basis parameters with unconverged k-points!

✅ **Start with 3×3 grid**: Quick test to identify optimal region

✅ **Use custodian for difficult systems**: Ionic compounds, metals, magnetic systems

✅ **Check basis size**: Note number of orbitals for each parameter combination

✅ **Material-specific ranges**: Ionic needs tighter than metals

✅ **Document your choice**: Save convergence plots and summary for your paper

✅ **Verify SCF convergence**: All calculations must converge properly

✅ **Balance accuracy vs. cost**: Tightest parameters not always necessary

---

## Next Steps

After completing basis parameter convergence:

1. **Use converged parameters** in production calculations
2. **Test different basis sizes**: DZ vs. DZP vs. TZP with optimal parameters
3. **Apply to workflows**: [EOS](../../02-equation-of-states/), [Elastic Constants](../../04-mechanical/), [Phonons](../../06-vibrational-properties/)
4. **Material-specific presets**: See [Tier System](../../../03-advanced-features/01-parameter-systems/01-tier-system/)

---

## Advanced: Custom Parameter Ranges

For advanced users wanting to explore specific regions:

```python
# Fine-grained scan around suspected optimum
energy_shifts = [0.008, 0.009, 0.010, 0.011, 0.012]  # 5 points around 0.01
split_norms = [0.13, 0.14, 0.15, 0.16, 0.17]          # 5 points around 0.15

# Very tight (for high accuracy calculations)
energy_shifts = [0.003, 0.005, 0.007, 0.010]
split_norms = [0.10, 0.12, 0.15]

# Very loose (for quick screening)
energy_shifts = [0.015, 0.020, 0.025, 0.030]
split_norms = [0.20, 0.25, 0.30]
```

---

## References

1. **SIESTA Manual**: Section on PAO basis generation
2. **Junquera et al.** (2001). "Numerical atomic orbitals for linear-scaling calculations." *Phys. Rev. B* 64, 235111.
3. **Artacho et al.** (2008). "The SIESTA method; developments and applicability." *J. Phys.: Condens. Matter* 20, 064208.

---

## Summary

**Key Takeaways**:
- PAO.EnergyShift and PAO.SplitNorm significantly affect accuracy and cost
- Silicon (covalent): Converges easily without custodian
- MgO (ionic): **Requires custodian** for automatic error recovery
- Optimal parameters: Usually ES ≈ 0.01 Ry, SN ≈ 0.15 (material-dependent)
- Always converge k-points and cutoff before testing basis parameters

**When to use custodian**:
- ✅ Ionic compounds (MgO, NaCl, oxides)
- ✅ Magnetic systems
- ✅ Difficult SCF convergence
- ✅ Production workflows (high success rate)

---

*Back to [Convergence Studies](../README.md) | [Main Tutorial Index](../../../README.md)*
