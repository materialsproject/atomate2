# EOS Basis Set Convergence Tutorial

**Category**: Advanced Workflows
**Difficulty**: Advanced
**Time**: ~10 min (dry-run), ~60-240 min (full study depending on configuration)
**Prerequisites**: Completed EOS tutorial (03-advanced-workflows/01-eos), understanding of basis sets (02-convergence/02-basis-parameters)

---

## Overview

This tutorial demonstrates systematic testing of **basis set effects on EOS parameters**. By running multiple Equation of State calculations with different basis sets and comparing the results, you can identify the optimal basis that balances accuracy and computational cost for your material.

### What You'll Learn

- Running parallel EOS calculations with different basis sets
- Comparing V₀, E₀, B₀ across basis quality levels
- Identifying convergence in EOS parameters
- Choosing optimal basis set for your material
- Understanding basis set effects on bulk properties
- Interpreting basis convergence plots

### Key Concepts

**Basis Set Convergence**: Process of testing increasingly sophisticated basis sets until properties no longer change significantly

**Why Test with EOS?**:
- Bulk properties (V₀, B₀) are sensitive to basis quality
- EOS provides multiple observables for convergence assessment
- More rigorous than single-point energy convergence
- Directly relevant to structural and mechanical properties

**Basis Set Hierarchy**:
```
SZ → DZ → DZP → DZDP → TZ → TZP
(fast, inaccurate) → (slow, accurate)
```

---

## Quick Start

### 1. Choose Example Type

Edit `tutorial.py` and set:

```python
# Choose example type: 'standard', 'comprehensive', or 'minimal'
EXAMPLE_TYPE = "standard"
```

**Available examples**:
- **`standard`**: 4 basis sets (SZ, DZ, DZP, DZDP) - recommended
- **`comprehensive`**: 6 basis sets including TZ and TZP - thorough
- **`minimal`**: 2 basis sets for quick testing (NOT for production)

### 2. Select Execution Mode

Uncomment ONE mode in `tutorial.py`:

```python
MODE = "dry_run"   # Preview structures
# MODE = "local"   # Run locally
# MODE = "submit"  # Submit to HPC
```

### 3. Run

```bash
cd tutorials/03-advanced-workflows/02-eos-basis-convergence
python tutorial.py
```

---

## Understanding Basis Set Effects

### Why Basis Sets Matter for EOS

**Basis set quality affects**:
1. **Equilibrium volume (V₀)**:
   - Underestimated with small basis
   - Converges with larger basis
   - Typical variation: 1-5%

2. **Bulk modulus (B₀)**:
   - Often overestimated with small basis
   - Very sensitive to basis quality
   - Typical variation: 5-15%

3. **Equilibrium energy (E₀)**:
   - Always too high (less negative) with small basis
   - Variational principle: E ≥ E_exact
   - Less relevant for property prediction

### Basis Set Hierarchy

**Single-Zeta (SZ)**:
- One radial function per orbital
- Fast but very inaccurate
- DO NOT use for production
- Useful only for workflow testing

**Double-Zeta (DZ)**:
- Two radial functions per orbital
- Significantly better than SZ
- Still lacks important flexibility
- Reasonable for quick screening

**Double-Zeta Polarized (DZP)**:
- DZ + polarization functions
- Good accuracy for most materials
- Recommended minimum for production
- Often sufficient for semiconductors/insulators

**Double-Double-Zeta Polarized (DZDP)**:
- Enhanced DZ basis + polarization
- High accuracy
- Good choice for demanding calculations
- Often converged for structural properties

**Triple-Zeta (TZ)**:
- Three radial functions
- Very high quality
- Expensive computationally
- Rarely needed

**Triple-Zeta Polarized (TZP)**:
- TZ + polarization
- Maximum accuracy in SIESTA
- Very expensive
- For benchmark calculations only

---

## Workflow Structure

### Calculation Steps

```
EOS Basis Convergence Workflow
├── For each basis set (parallel execution):
│   ├── Generate strained structures (5-9 volumes)
│   ├── Run relaxation/static calculations
│   ├── Fit E-V data to EOS models
│   └── Extract V₀, E₀, B₀, B₁
├── Collect results across all basis sets
├── Generate comparison plots:
│   ├── V₀ vs basis quality
│   ├── E₀ vs basis quality
│   ├── B₀ vs basis quality
│   └── Overlay of all E-V curves
└── Analyze convergence:
    ├── Calculate variation ranges
    ├── Identify convergence point
    └── Recommend optimal basis
```

### Parallel Execution

**Key advantage**: All basis set calculations are independent

```
Standard example (4 basis sets × 7 volumes = 28 calculations)
├── SZ:  7 calculations  ─┐
├── DZ:  7 calculations  ─┼─ All run in parallel
├── DZP: 7 calculations  ─┤
└── DZDP: 7 calculations ─┘
```

**Perfect for HPC clusters**: Submit all at once, wait for completion

---

## Example Configurations

### Standard Example (Recommended)

```python
EXAMPLE_TYPE = "standard"
```

**Configuration**:
- Basis sets: [SZ, DZ, DZP, DZDP]
- Strain range: -5% to +5%
- Volume points: 7 per basis
- Total calculations: 28
- Mesh cutoff: 300 Ry (fixed)
- K-points: 4×4×4 (fixed)

**When to use**:
- Most production convergence studies
- Understanding basis effects
- Choosing optimal basis for material
- Balancing accuracy vs cost

**Expected runtime**: ~60-90 minutes (local)

**What you learn**:
- Clear trend from SZ → DZDP
- Where convergence occurs
- Whether DZP is sufficient

### Comprehensive Example

```python
EXAMPLE_TYPE = "comprehensive"
```

**Configuration**:
- Basis sets: [SZ, DZ, DZP, DZDP, TZ, TZP]
- Strain range: -6% to +6%
- Volume points: 9 per basis
- Total calculations: 54
- Mesh cutoff: 350 Ry
- K-points: 6×6×6

**When to use**:
- Publication-quality studies
- Benchmark calculations
- Uncertain about DZDP convergence
- Maximum rigor required

**Expected runtime**: ~3-4 hours (local)

**What you learn**:
- Complete convergence profile
- Whether TZP provides improvement over DZDP
- Definitive basis recommendation

### Minimal Example

```python
EXAMPLE_TYPE = "minimal"
```

**Configuration**:
- Basis sets: [DZ, DZP]
- Strain range: -4% to +4%
- Volume points: 5 per basis
- Total calculations: 10
- Mesh cutoff: 200 Ry
- K-points: 3×3×3

**When to use**:
- Workflow testing ONLY
- Verifying setup before full study
- **NOT for production convergence**

**Expected runtime**: ~15-20 minutes (local)

---

## Output Files and Interpretation

### 1. eos_overlay_all_basis.png ⭐ **MOST USEFUL**

**Contents**: All EOS curves overlaid on single plot

**What it shows**:
- Scatter points: E-V data for each basis (different colors/markers)
- Fit curves: Birch-Murnaghan fits (smooth lines)
- Equilibrium points: Marked with X

**How to interpret**:

**Good convergence**:
```
┌──────────────────────┐
│    All curves overlap│
│    ⨯  ⨯  ⨯          │ ← Equilibrium points cluster
│   ● ● ● ●           │
│  ● ● ● ●            │
│ ● ● ● ●             │
│                      │
└──────────────────────┘
V (Ų) →
```
→ DZP already converged, no need for higher basis

**Poor convergence**:
```
┌──────────────────────┐
│  Curves spread apart │
│  ⨯   ⨯    ⨯         │ ← Equilibrium shifts
│ ●  ●  ●  ●           │
│●  ●  ●  ●            │
│                      │
└──────────────────────┘
V (Ų) →
```
→ Need higher basis sets to reach convergence

### 2. eos_basis_comparison.png

**Three-panel plot**:

**Top panel**: V₀ vs basis set
```
V₀ (Ų)
   │
43 │     ●────●────●    ← Converged
   │   ●
42 │ ●
   └────────────────
    SZ DZ DZP DZDP
```

**Middle panel**: E₀ vs basis set
```
E₀ (eV)
   │ ●────●────●────●
   │
   │ (Always decreasing with better basis)
   └────────────────
    SZ DZ DZP DZDP
```

**Bottom panel**: B₀ vs basis set
```
B₀ (GPa)
    │
100 │       ●────●    ← Converged
    │     ●
 90 │   ●
    └────────────────
     SZ DZ DZP DZDP
```

**Look for**: Plateau in top and bottom panels = convergence

### 3. eos_basis_summary.txt

**Contents**:

**Equilibrium Properties Table**:
```
================================================================================
Basis   V₀ (Ų)    E₀ (eV)     B₀ (GPa)   B₁
--------------------------------------------------------------------------------
SZ      42.123    -214.567    95.2       4.12
DZ      42.789    -215.234    97.8       4.05
DZP     43.012    -215.678    98.3       4.01
DZDP    43.045    -215.712    98.5       4.00
================================================================================
```

**Convergence Analysis**:
```
V₀ range: 0.033 Ų (0.08%)  ← Excellent (< 0.5%)
B₀ range: 0.2 GPa (0.20%)  ← Excellent (< 0.5%)
```

**Recommendation**:
```
DZP shows excellent convergence (< 0.5% change to DZDP)
RECOMMENDED: Use DZP for production calculations
```

---

## Convergence Criteria

### Quantitative Guidelines

**Excellent convergence** (< 0.5%):
```
ΔV₀ < 0.2 Ų    (for V₀ ~ 40 Ų)
ΔB₀ < 0.5 GPa  (for B₀ ~ 100 GPa)
```
→ Basis is fully converged, no improvement needed

**Good convergence** (< 1%):
```
ΔV₀ < 0.4 Ų
ΔB₀ < 1.0 GPa
```
→ Suitable for most production work

**Fair convergence** (< 2%):
```
ΔV₀ < 0.8 Ų
ΔB₀ < 2.0 GPa
```
→ Acceptable for screening, consider higher basis for final calculations

**Poor convergence** (> 2%):
```
ΔV₀ > 0.8 Ų
ΔB₀ > 2.0 GPa
```
→ Must use higher quality basis

### Typical Convergence Patterns

**Pattern 1: Early convergence** (most common):
```
SZ → DZ:    Large change (10-15%)
DZ → DZP:   Moderate (2-5%)
DZP → DZDP: Small (< 1%)     ← Converged
DZDP → TZP: Negligible (< 0.3%)
```
**Conclusion**: DZP or DZDP sufficient

**Pattern 2: Late convergence**:
```
SZ → DZ:    Large (10-15%)
DZ → DZP:   Moderate (5-8%)
DZP → DZDP: Moderate (2-3%)
DZDP → TZP: Small (< 1%)     ← Converged
```
**Conclusion**: Need DZDP or TZP

**Pattern 3: Never converges** (rare):
```
All transitions show > 2% changes
```
**Conclusion**: System has unusual electronic structure, may need different approach

---

## Material-Specific Guidelines

### Covalent Semiconductors (Si, Ge, GaAs)

**Typical finding**: Converge well with DZP

```
Recommended: DZP
Acceptable: DZDP (marginal improvement)
Overkill: TZP (< 0.3% change from DZDP)
```

**Example** (Si):
```
DZ → DZP:  ΔV₀ = 2.5%, ΔB₀ = 3.8%
DZP → DZDP: ΔV₀ = 0.4%, ΔB₀ = 0.6%  ← Converged
```

### Ionic Compounds (NaCl, MgO)

**Typical finding**: Also converge with DZP

```
Recommended: DZP
May need: DZDP for high accuracy
```

**Polarization functions critical** for ionic bonding

### Transition Metals (Fe, Cu, Ti)

**Typical finding**: Require DZDP for convergence

```
Minimum: DZP (screening only)
Recommended: DZDP
For magnetism: TZP may be needed
```

**Example** (Fe):
```
DZ → DZP:   ΔB₀ = 8%
DZP → DZDP: ΔB₀ = 2.5%
DZDP → TZP: ΔB₀ = 0.8%  ← Converged
```

### Molecules / Weak Interactions

**Typical finding**: Very sensitive to basis, need high quality

```
Minimum: DZDP
Recommended: TZP
For vdW: May need diffuse functions
```

---

## Best Practices

### 1. Keep Other Parameters Fixed

**Critical**: Only vary basis set, fix everything else

```python
# Same for ALL basis sets:
"Mesh.Cutoff": "300 Ry"  # Fixed
"kpts": [6, 6, 6]         # Fixed
"XC.functional": "GGA"    # Fixed
```

**Why**: Isolate basis set effects

### 2. Use Well-Converged Parameters

Before testing basis:
```bash
# Complete these first:
cd tutorials/02-convergence/01-kpoints-mesh-cutoff
python tutorial.py  # Find converged k-points and cutoff

# Then use those values for basis study
```

### 3. Test Multiple Properties

**Don't rely on V₀ alone**:
- V₀: Structural property
- B₀: Mechanical property
- B₁: Curvature (more sensitive)

All should converge together

### 4. Compare with Literature

**Silicon example**:
```
Experiment: V₀ = 20.0 Ų, B₀ = 99 GPa

Your results:
DZ:   V₀ = 19.5 Ų, B₀ = 95 GPa  (5% error)
DZP:  V₀ = 19.9 Ų, B₀ = 98 GPa  (1% error)
DZDP: V₀ = 20.0 Ų, B₀ = 98.5 GPa (0.5% error)

→ DZP already matches experiment well
→ DZDP gives marginal improvement
→ Choose DZP for production
```

### 5. Consider Computational Cost

**Scaling** (approximate, for same system):
```
SZ:   1×   (reference)
DZ:   3×
DZP:  5×
DZDP: 8×
TZ:   10×
TZP:  15×
```

**Choose basis where**:
```
Cost increase > Accuracy gain
```

**Example**: If DZP → DZDP gives < 1% improvement but costs 60% more time, stick with DZP

---

## Common Issues and Solutions

### Issue 1: "Different basis sets give very different EOS curves"

**Typical**: > 5% variation in B₀ between DZP and DZDP

**Causes**:
1. **Other parameters not converged** (most common)
   - Solution: Increase k-points to 8×8×8 or higher
   - Solution: Increase mesh cutoff to 400 Ry

2. **Initial structure not optimal**
   - Solution: Pre-relax with DZP before EOS

3. **System has unusual electronic structure**
   - Solution: May need higher basis inherently

### Issue 2: "Basis convergence is slower than expected"

**Symptoms**: Still seeing > 2% changes at DZDP level

**Materials prone to this**:
- Transition metals
- f-electron systems
- Systems with weak interactions

**Solutions**:
```python
# Test more basis sets
"basis_sets": ["DZ", "DZP", "DZDP", "TZ", "TZP"]

# Try diffuse basis (if available)
"PAO.SoftDefault": "True"
```

### Issue 3: "Results inconsistent with previous EOS tutorial"

**Symptoms**: Standard EOS with DZP doesn't match DZP result from convergence study

**Causes**:
- Different k-points or cutoff between studies
- Different strain range
- Different pseudopotentials

**Solution**: Ensure identical parameters

### Issue 4: "Some basis sets fail to complete"

**Symptoms**: TZP calculations fail while DZP succeeds

**Causes**:
- Memory exhaustion (TZP is large)
- SCF convergence issues

**Solutions**:
```python
# Increase memory allocation
# Or: Exclude TZP if only checking DZ-DZDP range

# Fix SCF issues
"MaxSCFIterations": 500
"SCF.Mixer.Weight": 0.1
```

### Issue 5: "Energy doesn't always decrease with better basis"

**Expected**: E₀ should become more negative with larger basis

**If not**:
- Check for SCF convergence failures
- Verify basis order is correct (SZ < DZ < DZP < DZDP)
- Could indicate numerical problems

---

## Advanced Topics

### Testing Basis Parameters

**Combine with basis parameter convergence**:

```python
# Test basis size AND PAO.EnergyShift together
from atomate2.siesta.flows.basis import BasisParametersConvergenceMaker

# For each basis size, also vary PAO.EnergyShift
```

See tutorial: `02-convergence/02-basis-parameters`

### Basis Set Superposition Error (BSSE)

**For molecules/weakly bound systems**:

```python
# Use counterpoise correction for BSSE
# Not yet implemented in atomate2siesta
# Consider ghost atoms manually
```

### Systematic Improvability

**SIESTA basis sets not variational** (unlike plane waves):
- Larger basis doesn't always give lower energy
- Convergence can be non-monotonic
- Use multiple properties for assessment

---

## Validation and Comparison

### Comparing with Plane-Wave Results

**SIESTA vs VASP/Quantum ESPRESSO**:

```
Well-converged SIESTA (DZP/DZDP):
  V₀: Within 1-2% of PW
  B₀: Within 5-10% of PW
  E₀: Absolute values differ (different pseudopotentials)
```

**Not expected to match exactly** - different basis philosophies

### Temperature Effects

**Remember**: DFT at 0 K, experiment typically at 300 K

**Thermal expansion corrections**:
```
ΔV/V ≈ 1-2% (300 K)
ΔB/B ≈ -5 to -10% (300 K)
```

Apply when comparing with experiment

---

## Summary Table: Basis Set Selection Guide

| Material Type | Minimum | Recommended | High Accuracy |
|--------------|---------|-------------|---------------|
| Semiconductors (Si, Ge) | DZ | DZP | DZDP |
| Ionic (NaCl, MgO) | DZ | DZP | DZDP |
| Metals (Al, Cu) | DZP | DZDP | TZP |
| Transition metals (Fe, Ti) | DZP | DZDP | TZP |
| Molecules | DZDP | TZP | TZP + diffuse |
| Weak interactions | DZDP | TZP | Custom |

---

## Next Steps

### After Completing This Tutorial

1. **Run standard example for your material**:
   - Test SZ, DZ, DZP, DZDP
   - Identify convergence point

2. **Validate convergence**:
   - Check V₀, B₀, B₁ all converge together
   - Verify < 1% change at chosen basis

3. **Compare with literature**:
   - Find experimental V₀ and B₀
   - Apply temperature corrections if needed

4. **Choose production basis**:
   - Use basis where properties are converged
   - Balance accuracy vs computational cost

5. **Apply to workflows**:
   - Use chosen basis for all subsequent calculations
   - Document choice in methods section

### Related Tutorials

- **02-convergence/02-basis-parameters**: Detailed basis parameter (PAO.EnergyShift, PAO.SplitNorm) optimization
- **03-advanced-workflows/01-eos**: Standard EOS calculation methodology
- **03-advanced-workflows/03-elastic-constants**: Elastic properties (also sensitive to basis)

---

## References

### SIESTA Basis Sets

1. **Artacho et al.** (2008). *J. Phys.: Condens. Matter* 20, 064208.
   - Numerical atomic orbitals in SIESTA
   - Basis set philosophy

2. **Soler et al.** (2002). *J. Phys.: Condens. Matter* 14, 2745.
   - Original SIESTA method paper
   - Basis set implementation

### Basis Set Convergence

1. **English et al.** (2015). *Comp. Mater. Sci.* 107, 113.
   - Systematic basis set studies
   - Convergence benchmarks

2. **Genovese et al.** (2011). *J. Phys.: Condens. Matter* 23, 085502.
   - Transferability of basis sets

---

## Summary

**What we covered**:
- ✅ Running parallel EOS calculations with multiple basis sets
- ✅ Comparing V₀, E₀, B₀ across basis quality
- ✅ Identifying convergence points quantitatively
- ✅ Interpreting overlay plots and convergence trends
- ✅ Material-specific basis recommendations
- ✅ Choosing optimal basis balancing accuracy vs cost

**Key takeaways**:
1. **Test multiple properties** (V₀, B₀, B₁) not just energy
2. **Converge other parameters first** (k-points, cutoff)
3. **DZP often sufficient** for semiconductors and insulators
4. **DZDP recommended** for transition metals and demanding applications
5. **TZP rarely needed** unless benchmarking or very high accuracy required
6. **Look for < 1% changes** between consecutive basis sets as convergence criterion

**Ready for**: Informed basis set selection, production calculations with optimal parameters, publication-quality computational methods sections

---

## Advanced: Full Parameter Optimization

For comprehensive parameter optimization, use `EOSFullBasisConvergenceMaker` which tests **all combinations** of basis sets AND PAO parameters.

### Files 04-05: Full Parameter Grid Search

**`04_full_parameter_optimization.py`**: Standard full optimization
- Tests: 2 basis × 3 energy_shifts × 2 split_norms = **12 EOS calculations**
- Each EOS: 5 volume points
- **Total**: 60 SIESTA runs
- **Purpose**: Find optimal PAO.EnergyShift and PAO.SplitNorm for each basis set

**`05_full_minimal.py`**: Quick test
- Tests: 1 basis × 2 energy_shifts × 2 split_norms = **4 EOS calculations**
- Minimal for workflow testing

### Comparison: Simple vs Full

**EOSBasisConvergenceMaker** (files 01-03):
```python
# Tests ONLY basis sets with FIXED parameters
basis_sets = ["DZ", "DZP", "TZP"]
# DZ uses ES=0.01, DZP uses ES=0.01, TZP uses ES=0.005 (predefined)
# → 3 EOS calculations
```

**EOSFullBasisConvergenceMaker** (files 04-05):
```python
# Tests ALL combinations
basis_sizes = ["DZ", "DZP"]
energy_shifts = [0.01, 0.015, 0.02]  # Test 3 values
split_norms = [0.15, 0.20]           # Test 2 values
# → 2 × 3 × 2 = 12 EOS calculations
```

### When to Use Full Optimization

**Use `EOSFullBasisConvergenceMaker` when**:
- You need to find optimal PAO parameters for each basis
- Publication requires parameter justification
- Working with unusual materials (actinides, heavy elements)
- Benchmarking against experiment or higher-level theory

**Use `EOSBasisConvergenceMaker` when**:
- Quick basis comparison with reasonable defaults
- Standard materials (semiconductors, simple metals)
- Just determining which basis size is sufficient

### Output Differences

Both workflows produce:
- `eos_overlay.png` - All EOS curves on one plot
- `eos_basis_comparison.png` - V₀, E₀, B₀ comparison
- `eos_basis_summary.txt` - Parameter recommendations

Full optimization ADDITIONALLY provides:
- Optimal PAO.EnergyShift for each basis
- Optimal PAO.SplitNorm for each basis
- Cost vs accuracy tradeoffs within each basis

---

## Tutorial Files

This directory contains three tutorials demonstrating different approaches to EOS basis convergence:

### Tutorial 01: Standard Usage

**`EOSBasisConvergenceFlowMaker_01_standard.py`**

Basic usage of EOSBasisConvergenceFlowMaker:
```python
from atomate2.siesta.flows.basis import EOSBasisConvergenceFlowMaker

flow = EOSBasisConvergenceFlowMaker(
    basis_sets=["SZ", "DZ", "DZP", "DZDP"],
    linear_strain=(-0.05, 0.05),
    number_of_frames=7,
)
workflow = flow.make(structure)
```

**When to use**: Quick basis convergence study with default parameters.

---

### Tutorial 02: Customizing with Powerups

**`EOSBasisConvergenceFlowMaker_02_powerups.py`**

Use powerups to customize parameters after workflow creation:
```python
from atomate2.siesta.flows.basis import EOSBasisConvergenceFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings

# Create workflow
maker = EOSBasisConvergenceFlowMaker(basis_sets=["SZ", "DZ", "DZP", "DZDP"])
workflow = maker.make(structure)

# Apply powerups to customize ALL jobs
workflow = update_user_siesta_settings(
    workflow,
    {
        "Mesh.Cutoff": "300 Ry",
        "a2s_kpts": [6, 6, 1],  # 2D material
        "SCF.Mixer.Weight": 0.05,
    },
)
```

**When to use**:
- Fine-tuning parameters for specific runs
- Adding parameters like Mesh.Cutoff, k-points, SCF settings
- Modifying existing workflows without recreating makers

**Key powerup functions**:
- `update_user_siesta_settings`: Update SIESTA FDF parameters
- `update_siesta_custodian_handlers`: Modify error handlers
- `set_dry_run`: Switch between dry-run and real calculations

---

### Tutorial 03: Using Tier Presets

**`EOSBasisConvergenceFlowMaker_03_presets.py`**

Apply production-quality parameter sets via tier presets:
```python
from atomate2.siesta.flows.basis import EOSBasisConvergenceFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings
from atomate2.siesta.sets.tiers import get_tier_preset

# Get preset configuration
preset = get_tier_preset("2d_semiconductor")

# Create workflow
maker = EOSBasisConvergenceFlowMaker(basis_sets=["SZ", "DZ", "DZP", "DZDP"])
workflow = maker.make(structure)

# Apply preset parameters with custom overrides
preset_params = preset["recommended_params"].copy()
preset_params["Mesh.Cutoff"] = "350 Ry"  # Custom override
workflow = update_user_siesta_settings(workflow, preset_params)
```

**When to use**:
- Production calculations requiring validated parameter sets
- Material-specific settings (2D materials, surfaces, magnetic systems)
- Reproducible, documented workflows

**Available presets**:
- `relax_standard`: Standard relaxation (mesh_cutoff=200 Ry, kpts=[4,4,4])
- `relax_high_accuracy`: High-accuracy relaxation
- `2d_semiconductor`: Optimized for 2D semiconductors (MoS2, etc.)
- `2d_metal`: Optimized for 2D metals (graphene, etc.)
- `surface_metal`: Optimized for metallic surfaces
- `magnetic_*`: Optimized for magnetic systems

---

## Comparison: Three Methods

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Standard** | Quick testing | Simple, minimal code | No parameter customization |
| **Powerups** | Fine-tuning | Flexible, modify after creation | Requires workflow object |
| **Presets** | Production | Validated params, reproducible | Need to know preset names |

### Recommendation

1. **Start with Standard** (Tutorial 01) to understand the workflow
2. **Use Powerups** (Tutorial 02) for parameter exploration
3. **Use Presets** (Tutorial 03) for production calculations

---

*Tutorial created: 2024-10-22*
*Last updated: 2026-01-23*
*Back to [Advanced Workflows](../README.md) | [Main Tutorial Index](../../README.md)*
