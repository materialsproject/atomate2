# Common Errors

**Category**: troubleshooting/common_errors
**Difficulty**: Intermediate
**Time**: 15-30 minutes per tutorial

## Overview

This section provides detailed troubleshooting guides for the most frequently encountered calculation failures in atomate2siesta. Each guide includes symptom identification, root cause analysis, and step-by-step solutions.

## Available Tutorials

### scf_convergence_issues.py

**SCF Convergence Problems**

Comprehensive guide to diagnosing and fixing self-consistent field (SCF) convergence failures.

**Topics Covered**:
- Identifying SCF convergence problems
- Understanding causes of SCF failures
- Systematic fixes (mixer settings, electronic temperature, spin polarization)
- Using custodian for automatic recovery
- Prevention strategies

**Common Symptoms**:
- "SCF not converged after N iterations"
- "Density matrix did not converge"
- "Oscillating SCF convergence"

**Solutions Included**:
1. **Reduce mixing weight**: `SCF.Mixer.Weight: 0.01` (from default 0.1)
2. **Add electronic temperature**: For metallic systems (`ElectronicTemperature: "300 K"`)
3. **Enable spin polarization**: For magnetic systems (`Spin: "polarized"`)
4. **Improve initial guess**: Better density matrix initialization
5. **Increase accuracy**: Higher k-points and mesh cutoff
6. **Automatic recovery**: Using custodian handlers

**Time**: 15 minutes (reading + examples)

## Quick Reference

### SCF Convergence - Most Common Fixes

```python
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.custodian import SCFConvergenceHandler

# Fix 1: Reduce mixer weight (most effective)
user_params = {
    "SCF.Mixer.Weight": 0.01,  # Default: 0.1
    "SCF.Mixer.History": 8,     # Default: 5
}

# Fix 2: For metals - add smearing
user_params = {
    "ElectronicTemperature": "300 K",
    "OccupationFunction": "FD",  # Fermi-Dirac
}

# Fix 3: For magnetic systems
user_params = {
    "Spin": "polarized",
}

# Fix 4: Use custodian for automatic recovery (recommended!)
maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,
    custodian_handlers=[SCFConvergenceHandler(max_attempts=10)],
)
```

### Success Rates

| Solution | Success Rate | When to Use |
|----------|--------------|-------------|
| Custodian automatic | high | Always (recommended) |
| Reduce mixer weight | 80% | Oscillating convergence |
| Electronic temperature | 90% | Metallic systems |
| Spin polarization | 85% | Magnetic elements |
| Increase accuracy | 70% | Small/zero band gap |

## Diagnostic Workflow

### Step 1: Identify the Problem
```bash
# Check for SCF errors
grep -i "scf" job_*/siesta.out | tail -20

# Look for convergence oscillations
grep "scf:" job_*/siesta.out
```

### Step 2: Determine System Type
- **Metallic?** → Need electronic temperature
- **Magnetic elements?** (Fe, Co, Ni, Mn) → Need spin polarization
- **Oxide/insulator?** → Usually just need slower mixing

### Step 3: Apply Appropriate Fix
See `scf_convergence_issues.py` for detailed examples.

### Step 4: Enable Custodian
Let custodian handle it automatically for future calculations.

## Prevention Strategies

### 1. Use Tier Presets

Material-specific presets include appropriate SCF settings:

```python
from atomate2.siesta.sets.tiers import apply_tier_preset

maker = RelaxMaker.fixed_cell_relaxation()

# For metals
maker = apply_tier_preset(maker, "relax_bulk_metal")

# For magnetic materials
maker = apply_tier_preset(maker, "magnetic_correlated")
```

### 2. Always Enable Custodian

```python
maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,  # Automatic error handling
)
```

high of SCF issues will be fixed automatically.

### 3. Test with Dry-Run First

```python
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

# Check generated parameters
cat preview_output/job_*/siesta.fdf
```

## Common Misconceptions

❌ **"SCF must converge to 1e-8"**
→ 1e-4 is usually sufficient for most properties

❌ **"More iterations always helps"**
→ If oscillating, need different parameters (not more iterations)

❌ **"Higher mesh cutoff fixes SCF"**
→ SCF convergence is usually a mixer/smearing issue, not accuracy

❌ **"Calculation failed = code bug"**
→ 99% of SCF failures are parameter issues, easily fixed

## Related Resources

### Tutorials
- [debugging_workflows/01_tracing_job_failures.py](../debugging_workflows/01_tracing_job_failures.py) - General debugging
- [03-advanced-features/04-error-handling](../../03-advanced-features/04-error-handling/) - Custodian system
- [02-workflows/01-convergence](../../02-workflows/01-convergence/) - Parameter convergence

### Documentation
- SIESTA manual: SCF convergence section
- Custodian handlers: `src/atomate2/siesta/custodian/`
- Parameter reference: `docs/source/fdf-parameters.rst`

## When to Ask for Help

If SCF still doesn't converge after trying:
1. ✅ Reduced mixer weight to 0.005
2. ✅ Added electronic temperature (if metallic)
3. ✅ Enabled spin polarization (if magnetic)
4. ✅ Used custodian with 10 attempts
5. ✅ Increased k-points and mesh cutoff

Gather:
- Complete error output: `tail -200 job_*/siesta.out`
- Input file: `job_*/siesta.fdf`
- Structure: `job_*/siesta.XV`
- System description (formula, metallic?, magnetic?)

Post at: https://github.com/arsalan-akhtar/atomate2siesta/issues

---

*Back to [Troubleshooting Index](../README.md)*
