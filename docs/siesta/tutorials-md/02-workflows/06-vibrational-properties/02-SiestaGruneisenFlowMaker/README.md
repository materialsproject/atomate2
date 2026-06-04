# Tutorial: Grüneisen Parameters

**Category**: 05-vibrational-properties
**Difficulty**: Advanced
**Time**: ~5 min (dry-run), ~2-3 hours (full calculation)

---

## Overview

Calculate mode-dependent Grüneisen parameters to characterize how phonon frequencies change with volume. Essential for understanding thermal expansion and anharmonicity in materials.

This consolidates 8 example scripts into a single configurable workflow with 5 comprehensive examples and 6 plotting functions.

---

## What You'll Learn

- Mode-dependent Grüneisen parameters γᵢ
- Average Grüneisen parameter γ̄
- Thermal expansion coefficient α(T) calculation
- Finite difference method (phonons at 3 volumes)
- Anharmonicity characterization
- Volume perturbation selection
- Complete visualization suite (6 plotting functions)

---

## Prerequisites

- **Required**: [16-phonon-calculations](../01-phonon-calculations/) completed
- **Recommended**: Understanding of phonon dispersion
- **Recommended**: Basic thermodynamics knowledge

---

## Key Concepts

### Grüneisen Parameter

$$
\gamma_i = -\frac{V}{\omega_i} \frac{\partial \omega_i}{\partial V}
$$

**Physical Interpretation**:
- **γᵢ > 0**: Frequency decreases with expansion (most common)
- **γᵢ < 0**: Frequency increases with expansion (rare, unusual bonding)
- **|γᵢ| large**: Strongly anharmonic mode
- **|γᵢ| small**: Weakly anharmonic mode

### Thermal Expansion

$$
\alpha = \frac{\bar{\gamma} \cdot C_V}{B \cdot V}
$$

where γ̄ is the average Grüneisen parameter, Cᵥ is heat capacity, B is bulk modulus, V is volume.

### Computational Method

```
1. Phonon at V₋ = V₀(1 - δ)   (compressed)
2. Phonon at V₀                (equilibrium)
3. Phonon at V₊ = V₀(1 + δ)   (expanded)

4. Compute: γᵢ = -(V₀/ωᵢ) · [ωᵢ(V₊) - ωᵢ(V₋)] / [V₊ - V₋]
```

**Volume Perturbation δ**:
- Normal materials: 0.01 (±1%)
- Hard materials: 0.02 (±2%)
- Soft materials: 0.005 (±0.5%)

---

## Configuration Options

### Example Types

#### 1. Basic (Silicon)
```python
EXAMPLE_TYPE = "basic"
```
- Standard ±1% volume perturbation
- 2×2×2 supercell for speed
- Learn basic workflow
- Expected γ̄ ≈ 1.0

#### 2. Custom Volume (Diamond)
```python
EXAMPLE_TYPE = "custom_volume"
```
- Larger ±2% perturbation (hard material)
- 3×3×3 supercell for accuracy
- Bulk modulus ~440 GPa
- Expected γ̄ ≈ 0.5-1.0

#### 3. Optimization (NaCl)
```python
EXAMPLE_TYPE = "optimization"
```
- Full structure optimization at each volume
- Ensures consistent forces
- More accurate but expensive
- Variable-cell relaxation

#### 4. High Accuracy (MgO)
```python
EXAMPLE_TYPE = "high_accuracy"
```
- TZP basis, 500 Ry cutoff
- 8×8×8 k-points
- 4×4×4 supercell (128 atoms)
- Publication-quality settings

#### 5. Plotting (Silicon with 6 plots)
```python
EXAMPLE_TYPE = "plotting"
```
- Complete analysis workflow
- 6 plotting functions
- Thermal expansion α(T)
- Comprehensive summary

---

## Quick Start

```bash
# 1. Preview workflow
# Edit tutorial.py: EXAMPLE_TYPE = "basic", RUN_MODE = "dry_run"
python tutorial.py

# 2. Inspect workflow structure
ls preview_output/job_*
# Should see: 3 phonon calculations (V₋, V₀, V₊)

# 3. Try plotting example (dry-run first!)
# Edit tutorial.py: EXAMPLE_TYPE = "plotting"
python tutorial.py

# 4. Run calculation (WARNING: ~2-3 hours)
# Edit tutorial.py: RUN_MODE = "local"
python tutorial.py
```

---

## Expected Output

### Dry-Run Mode

```
✅ Dry-run complete!

📁 Workflow Structure:
  1. Phonon at V₀×0.99 (compressed)
  2. Phonon at V₀ (equilibrium)
  3. Phonon at V₀×1.01 (expanded)
  4. Grüneisen Analysis (automatic)

💡 Total jobs: 3 phonon calculations
```

### Local Mode

```
✅ Grüneisen calculation complete!

📊 Output includes:
  - Mode-dependent γᵢ
  - Average γ̄
  - Thermal expansion estimate
  - Anharmonicity indicators
```

**If plotting example**:
```
✅ All plots saved in 'gruneisen_plots/' directory:
  - gruneisen_vs_frequency.png
  - gruneisen_distribution.png
  - gruneisen_band_structure.png (if available)
  - thermal_expansion.png
  - gruneisen_summary.txt
```

---

## Plotting Functions

The tutorial includes 6 comprehensive plotting functions:

### 1. plot_gruneisen_vs_frequency()
- Scatter plot: γᵢ vs. ωᵢ
- Color-coded by Grüneisen value
- Statistics overlay
- Identifies mode trends

### 2. plot_gruneisen_distribution()
- Histogram of γᵢ values
- Mean and median markers
- Distribution shape analysis
- Color-coded bars

### 3. plot_gruneisen_band_structure()
- Dual-panel plot
- Phonon bands with γᵢ color
- Mode-specific anharmonicity
- High-symmetry path

### 4. calculate_thermal_expansion()
- α(T) using α = γ̄ Cᵥ / (B·V)
- Temperature-dependent (0-1000 K)
- Requires bulk modulus
- Returns T and α arrays

### 5. plot_thermal_expansion()
- α(T) vs. temperature
- Comparison with experiment
- Units: 10⁻⁶ K⁻¹
- Temperature dependence

### 6. write_gruneisen_summary()
- Comprehensive text report
- Structure information
- Statistics and interpretation
- Convergence recommendations

---

## Analyzing Results

### Typical Ranges

**By Material Type**:
- Soft materials (Na, K): γ̄ ≈ 1.5-3.0
- Normal materials (Si, Al, Cu): γ̄ ≈ 1.0-2.0
- Hard materials (MgO, Al₂O₃): γ̄ ≈ 0.5-1.5
- Very hard (Diamond, SiC): γ̄ < 1.0

**Mode-Dependent**:
- Acoustic modes: typically γ ≈ 1-2
- Optical modes: wide range γ ≈ 0-4
- Negative γ: rare (check carefully!)

### Validation Checks

✓ **Most γᵢ positive**: 0 < γ < 3 typical
✓ **Average γ̄ reasonable**: Compare with literature
✓ **Few negative γ**: < 10% of modes (if any)
✓ **No very large |γᵢ|**: > 5 suggests convergence issues
✓ **Thermal expansion**: α(300K) matches experiment

### Material Classification

```
Material      | γ̄   | Interpretation
--------------|-----|------------------
Diamond       | 0.8 | Very stiff, weak anharmonicity
Silicon       | 1.0 | Normal, moderate anharmonicity
Aluminum      | 2.1 | Soft, strong anharmonicity
Sodium        | 2.5 | Very soft, very anharmonic
```

---

## Volume Perturbation Selection

### Guidelines by Bulk Modulus

| Material | Bulk Modulus | δ (%) | Example |
|----------|--------------|-------|---------|
| Soft | < 50 GPa | 0.5-1 | Na, K |
| Normal | 50-150 GPa | 1 | Al, Si, Cu |
| Hard | 150-300 GPa | 1-2 | MgO, Al₂O₃ |
| Very Hard | > 300 GPa | 2 | Diamond |

**Rule**: Harder materials → Larger perturbation (better numerical accuracy)

### Testing Different δ

```python
# Test δ = 0.01 vs 0.02
maker_1 = SiestaGruneisenMaker(perc_vol=0.01)
maker_2 = SiestaGruneisenMaker(perc_vol=0.02)

# Compare results
# If γ̄ differs by > 10%, increase supercell or k-points
```

---

## Common Issues

### Issue 1: Many Negative γᵢ

**Symptoms**: > 20% modes with γᵢ < 0

**Solutions**:
1. Optimize structure before phonons
2. Check phonon stability (no imaginary modes)
3. Increase δ (0.01 → 0.02)
4. Verify k-point convergence
5. May be genuine (check literature)

### Issue 2: Very Large |γᵢ| > 10

**Symptoms**: Some modes unreasonably large

**Solutions**:
1. Increase supercell size (2×2×2 → 3×3×3)
2. Check ∂ω/∂V numerical stability
3. Increase δ for hard materials
4. Tighten SCF (DM.Tolerance < 1e-6)
5. May indicate soft mode (phase transition)

### Issue 3: Unphysical Thermal Expansion

**Symptoms**: α(T) negative or >> experiment

**Solutions**:
1. Check bulk modulus (use EOS, not estimate!)
2. Verify γ̄ is reasonable
3. Compare with literature γ̄
4. Check phonon convergence
5. QHA valid only T < 0.6 T_melt

### Issue 4: Inconsistent Results

**Symptoms**: Different δ gives very different γ

**Solutions**:
1. Ensure phonon convergence at all volumes
2. Check force convergence (< 0.01 eV/Å)
3. Increase k-point density
4. Verify cutoff convergence
5. Try intermediate δ

---

## Convergence Tests

Always test convergence of:

1. ✅ **Supercell size**: 2×2×2 vs 3×3×3 vs 4×4×4
2. ✅ **K-point mesh**: [2,2,2] vs [4,4,4] vs [6,6,6]
3. ✅ **Volume perturbation**: δ=0.01 vs 0.02
4. ✅ **Cutoff energy**: 200 vs 300 vs 500 Ry
5. ✅ **Compare with QHA**: If available

**Convergence Criterion**: γ̄ changes by < 5% when increasing parameter

---

## Best Practices

**Parameter Selection**:
- Testing: 2×2×2 supercell, [2,2,2] k-points, δ=0.01
- Standard: 3×3×3 supercell, [4,4,4] k-points, δ=0.01
- High accuracy: 4×4×4 supercell, [6,6,6] k-points, δ=0.02

**Workflow Design**:
1. Dry-run first with basic settings
2. Test convergence systematically
3. Production run with converged settings
4. Use plotting example for complete analysis

**Resource Management**:
- Total time = 3 × phonon time
- Si (2×2×2): ~1-1.5 hours
- MgO (3×3×3): ~3-6 hours
- Diamond (4×4×4): ~10-15 hours

---

## Advanced Topics

### Comparison with QHA

**Grüneisen Method**:
- ✓ Faster (3 phonon calculations vs 5-7 for QHA)
- ✓ Direct mode anharmonicity
- ✗ Approximate thermal expansion (needs bulk modulus)

**QHA**:
- ✓ More accurate thermal expansion
- ✓ Full thermodynamic properties
- ✗ More expensive (5-7 volumes)

**Best Practice**: Use both for validation!

### Temperature-Dependent Properties

From Grüneisen parameters:
```python
# Calculate α(T)
thermal_props = calculate_thermal_expansion(
    gruneisen_doc=doc,
    bulk_modulus=100.0,  # GPa, from EOS
    temperatures=range(0, 1001, 10)  # 0-1000 K
)

# Plot α(T)
plot_thermal_expansion(thermal_props)
```

### High-Pressure Applications

Predict frequency shifts under pressure:

$$
\omega(P) = \omega_0 \exp\left(\gamma \ln\frac{V_0}{V(P)}\right)
$$

---

## Tips for Success

✅ **Start with basic example**: Test workflow first
✅ **Harder materials → larger δ**: ±2% for diamond, ±1% for silicon
✅ **Check phonon stability**: No imaginary modes at any volume
✅ **Most γᵢ should be positive**: Negative γ rare
✅ **Use plotting example**: Complete analysis workflow
✅ **Compare with literature**: Validate γ̄ and α
✅ **Combine with EOS**: Get accurate bulk modulus
✅ **Test convergence**: Supercell, k-points, δ

---

## Next Steps

After completing this tutorial:

1. **Understand workflow**: Review dry-run output
2. **Test convergence**: Vary supercell, k-points, δ
3. **Try all examples**: basic → custom → optimization → high_accuracy → plotting
4. **Validate**: Compare γ̄ with literature
5. **Advanced**: Combine with QHA (tutorial 21)
6. **Applications**: Predict thermal expansion, study anharmonicity

---

*Back to [05-vibrational-properties](../README.md) | [Main Tutorial Index](../../README.md)*
