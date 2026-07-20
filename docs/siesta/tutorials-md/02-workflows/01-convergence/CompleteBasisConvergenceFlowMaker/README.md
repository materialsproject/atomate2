# Tutorial: Complete Basis Convergence

**Category**: 02-convergence
**Difficulty**: Advanced
**Time**: ~1-4 hours (depending on configuration)

---

## Overview

Comprehensive basis convergence study that tests BOTH basis sizes (SZ, DZ, DZP, TZP) AND basis parameters (PAO.EnergyShift, PAO.SplitNorm) simultaneously. This workflow finds the optimal combination of basis quality and parameter settings.

---

## Examples in This Tutorial

1. **`01_simple_complete.py`** - Simple test (2×2×2 = 8 calculations)
2. **`02_full_complete.py`** - Full study (3×3×3 = 27 calculations)
3. **`03_with_custodian.py`** - With automatic error handling (3×2×2 = 12 calculations)

---

## What You'll Learn

- Simultaneous basis size + parameter optimization
- CompleteBasisConvergenceMaker workflow
- Comprehensive convergence analysis
- Cost vs. accuracy trade-offs across multiple dimensions
- Production-ready basis selection

---

## When to Use This Workflow

### Use Complete Basis Convergence When:

✅ Starting a new material study from scratch
✅ Need to establish production settings for a material class
✅ Want to understand the full convergence landscape
✅ Have computational resources for comprehensive testing
✅ Publication-quality accuracy is required

### Use Separate Workflows When:

- **Basis Parameters Only** (Tutorial 02): Already know which basis size to use
- **Basis Size Only**: Already have optimized parameters, just testing DZ vs DZP vs TZP

---

## Workflow Architecture

### What CompleteBasisConvergenceMaker Does

```
For each basis size (DZ, DZP, TZP):
  ├─ For each PAO.EnergyShift (0.01, 0.015, 0.02 Ry):
  │   ├─ For each PAO.SplitNorm (0.15, 0.20, 0.25):
  │   │   └─ Run SCF calculation
  │   └─ Analyze parameter convergence for this basis size
  └─ Compare across basis sizes
```

**Total calculations**: `n_basis × n_shifts × n_norms`

---

## Configuration Examples

### 1. Simple Test (`01_simple_complete.py`)

```python
maker = CompleteBasisConvergenceMaker(
    basis_sizes=["DZ", "DZP"],       # 2 basis sizes
    energy_shifts=[0.01, 0.02],      # 2 energy shifts
    split_norms=[0.15, 0.20],        # 2 split norms
    kpts=[3, 3, 3],
)
# Total: 2 × 2 × 2 = 8 calculations (~30 min)
```

### 2. Full Study (`02_full_complete.py`)

```python
maker = CompleteBasisConvergenceMaker(
    basis_sizes=["DZ", "DZP", "TZP"],  # 3 basis sizes
    energy_shifts=[0.005, 0.01, 0.015], # 3 energy shifts
    split_norms=[0.15, 0.20, 0.25],     # 3 split norms
    kpts=[4, 4, 4],
)
# Total: 3 × 3 × 3 = 27 calculations (~1-2 hours)
```

### 3. With Custodian (`03_with_custodian.py`)

```python
maker = CompleteBasisConvergenceMaker(
    basis_sizes=["DZ", "DZP", "TZP"],
    energy_shifts=[0.01, 0.02],
    split_norms=[0.15, 0.20],
    kpts=[3, 3, 3],
    use_custodian=True,              # Automatic error handling
    custodian_max_errors=10,
)
# Total: 3 × 2 × 2 = 12 calculations with recovery
```

---

## Expected Output

### 1. Convergence Analysis

```
✅ Complete Basis Convergence Results

Tested configurations:
  • 3 basis sizes: DZ, DZP, TZP
  • 3 energy shifts: 0.005, 0.01, 0.015 Ry
  • 3 split norms: 0.15, 0.20, 0.25
  • Total: 27 calculations

Optimal configuration:
  Basis size: DZP
  PAO.EnergyShift: 0.010 Ry (136 meV)
  PAO.SplitNorm: 0.20
  Total energy: -214.523456 eV
  Number of orbitals: 156

Convergence criteria:
  ✓ Energy stable within 2 meV
  ✓ Basis size reasonable (< 500 orbitals)
  ✓ Next level (TZP) gives < 5 meV improvement
```

### 2. Visualization

**Generated files**:
- `complete_convergence_by_basis.png` - Energy vs parameters for each basis size
- `complete_convergence_comparison.png` - Basis size comparison
- `basis_size_vs_energy.png` - Overall convergence trend
- `complete_summary.txt` - Detailed analysis and recommendations

---

## Comparison with Other Tutorials

| Tutorial | What It Tests | Total Jobs | Time | Use Case |
|----------|---------------|------------|------|----------|
| **02-basis-parameters** | Parameters only (fixed basis) | 9-25 | 30 min - 1 hr | Known basis size |
| **03-complete-basis** | Sizes + Parameters | 8-27+ | 1-4 hours | Full optimization |

---

## Material-Specific Recommendations

### Covalent Systems (Si, Diamond)

```python
basis_sizes = ["DZ", "DZP", "TZP"]
energy_shifts = [0.01, 0.012, 0.015, 0.018, 0.02]
split_norms = [0.12, 0.15, 0.18, 0.20]
```

**Expected optimal**: DZP with EnergyShift ≈ 0.01-0.015 Ry

### Ionic Systems (MgO, NaCl)

```python
basis_sizes = ["DZ", "DZP", "DZDP"]
energy_shifts = [0.005, 0.008, 0.01, 0.012]
split_norms = [0.15, 0.18, 0.20, 0.22]
```

**Expected optimal**: DZP with EnergyShift ≈ 0.005-0.01 Ry (tighter)

### Metallic Systems (Al, Cu)

```python
basis_sizes = ["DZ", "DZP"]  # Often DZ sufficient
energy_shifts = [0.015, 0.018, 0.02, 0.022]
split_norms = [0.15, 0.18, 0.20]
```

**Expected optimal**: DZ or DZP with EnergyShift ≈ 0.015-0.02 Ry (looser)

---

## Best Practices

### Workflow Strategy

1. **Start simple** (2×2×2 = 8 calcs) to identify region
2. **Refine if needed** with denser grid in optimal region
3. **Verify convergence** by checking next level improvement
4. **Document settings** for reproducibility

### Computational Efficiency

**Grid design**:
- Start with 2-3 values per dimension (8-27 calculations)
- Focus on realistic parameter ranges for your material
- Use coarser k-points for initial scan (3×3×3)
- Refine k-points once basis is optimized

**Time management**:
- Simple: 2×2×2 grid = ~30 minutes
- Medium: 3×2×2 grid = ~1 hour
- Full: 3×3×3 grid = ~2-3 hours
- Comprehensive: 4×4×4 grid = ~8-12 hours

---

## Interpreting Results

### Energy Convergence Criteria

**Well converged**:
- Next basis level gives < 5 meV improvement
- Parameters show flat energy landscape
- Basis size < 500 orbitals per atom

**Needs refinement**:
- Energy still decreasing with better basis
- Large jumps between parameter values
- Basis size excessive (> 1000 orbitals)

### Cost vs. Accuracy Trade-off

```
Basis progression:
DZ → DZP:  +50% cost, +10-20 meV accuracy
DZP → DZDP: +100% cost, +5-10 meV accuracy
DZDP → TZP: +150% cost, +2-5 meV accuracy

Typical choice: DZP (best balance)
High accuracy: DZDP or TZP
```

---

## Common Issues

### Issue 1: Too Many Calculations

**Problem**: 27+ calculations take too long

**Solution**: Start with fewer test points
```python
# Instead of 3×3×3 = 27
basis_sizes = ["DZ", "DZP"]      # 2 instead of 3
energy_shifts = [0.01, 0.015]     # 2 instead of 3
split_norms = [0.15, 0.20]        # 2 instead of 3
# Now: 2×2×2 = 8 calculations
```

### Issue 2: Results Are Inconsistent

**Problem**: Energy landscape is noisy

**Solutions**:
1. **First converge k-points** (Tutorial 01)
2. **Increase MaxSCFIterations** to 200
3. **Tighten DM.Tolerance** to 1e-5
4. **Use custodian** for automatic recovery

### Issue 3: Optimal Point at Edge

**Problem**: Best parameters at edge of tested range

**Solution**: Extend range in that direction
```python
# If optimal at lowest EnergyShift:
energy_shifts = [0.003, 0.005, 0.008, 0.01]  # Extend lower

# If optimal at highest SplitNorm:
split_norms = [0.20, 0.25, 0.30, 0.35]  # Extend higher
```

---

## Advanced Topics

### Combining with k-point Convergence

For production settings, iterate:
1. Rough k-point convergence with DZ basis
2. Complete basis convergence at converged k-points
3. Final k-point refinement with optimal basis

### Force Accuracy Requirements

For geometry optimization, phonons, NEB:
- Need tighter basis than for energies alone
- Test: Minimal should be DZP + EnergyShift ≤ 0.01 Ry
- Verify forces converge to < 0.001 eV/Å

### System Size Considerations

**Small systems** (< 20 atoms):
- Can afford TZP or DZDP
- Tight parameters (EnergyShift = 0.005 Ry)

**Medium systems** (20-100 atoms):
- DZP or DZDP practical
- Standard parameters (EnergyShift = 0.01 Ry)

**Large systems** (> 100 atoms):
- DZP maximum, often DZ sufficient
- Looser parameters (EnergyShift = 0.015-0.02 Ry)

---

## Tips for Success

✅ **Prerequisites**: Complete Tutorial 01 (k-points) first

✅ **Start small**: 2×2×2 grid for quick test

✅ **Use custodian**: Prevent wasted time on SCF failures

✅ **Material-specific**: Consult guidelines above

✅ **Check basis size**: Note number of orbitals in output

✅ **Verify SCF**: All calculations must converge

✅ **Document choice**: Record optimal settings for future

✅ **Compare literature**: When data available

---

## Next Steps

1. **Run simple example**: Quick 2×2×2 test
2. **Examine results**: Identify optimal region
3. **Scale up if needed**: Run full 3×3×3 study
4. **Apply to your material**: Adjust ranges appropriately
5. **Use optimized settings**: For all production calculations

---

## See Also

- **Tutorial 02** (`02-basis-parameters/`): Parameters only (faster)
- **Tutorial 01** (`01-kpoints-cutoff/`): K-point convergence
- **Advanced**: `03-advanced-workflows/02-eos-basis-convergence/` - EOS with basis

---

## References

1. **Junquera et al.** (2001). "Numerical atomic orbitals for linear-scaling calculations." *Phys. Rev. B* 64, 235111.
2. **SIESTA Manual**: Sections on basis sets and PAO parameters
3. **Artacho et al.** (2008). "The SIESTA method; developments and applicability." *J. Phys.: Condens. Matter* 20, 064208.

---

*Back to [02-convergence](../README.md) | [Main Tutorial Index](../../README.md)*
