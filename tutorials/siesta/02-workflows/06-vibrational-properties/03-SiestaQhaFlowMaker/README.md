# Tutorial: Quasi-Harmonic Approximation (QHA) - Temperature-Dependent Thermodynamics

**Category**: 05-vibrational-properties
**Difficulty**: Advanced
**Time**: ~10 min (dry-run), ~3-5 hours (full calculation)

---

## Overview

Calculate temperature-dependent thermodynamic properties using the Quasi-Harmonic Approximation (QHA). QHA combines phonon calculations with equation of state (EOS) to predict thermal expansion, heat capacity, and free energies as functions of temperature and pressure.

This tutorial consolidates 5 comprehensive QHA examples into a single configurable workflow, demonstrating basic calculations, high-pressure studies, custom accuracy settings, metallic systems, and wide temperature ranges.

---

## What You'll Learn

- QHA computational workflow (EOS + phonons at multiple volumes)
- Temperature-dependent properties (α, Cv, Cp, G, S, B, V)
- Pressure-dependent calculations for P-T diagrams
- High-accuracy settings for different material types
- Handling imaginary modes in metals
- Result analysis and validation
- QHA limitations and best practices

---

## Prerequisites

- **Required**: [16-phonon-calculations](../01-SiestaPhononFlowMaker/) completed
- **Required**: [09-equation-of-state](../../../02-workflows/02-equation-of-states/)
- **Recommended**: Understanding of statistical thermodynamics
- **Recommended**: Familiarity with phonon density of states

---

## Key Concepts

### What is QHA?

The **Quasi-Harmonic Approximation** combines:
1. **Equation of State**: E(V) relationship
2. **Phonons at Multiple Volumes**: ω(q,V) frequencies change with volume
3. **Statistical Mechanics**: Free energy from phonon DOS

**Key Assumption**: Phonons are harmonic at each fixed volume, but frequencies vary with volume (volume-dependent anharmonicity).

**NOT Included in QHA**:
- ❌ Phonon-phonon scattering (need phono3py)
- ❌ Temperature-dependent phonon linewidths
- ❌ Intrinsic anharmonicity at fixed volume

### Physical Properties from QHA

$$
\alpha(T) = \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_P
$$

**Thermal expansion coefficient**

$$
C_V(T) = \left(\frac{\partial E}{\partial T}\right)_V
$$

**Heat capacity (constant volume)**

$$
C_P(T) = \left(\frac{\partial H}{\partial T}\right)_P = C_V + \alpha^2 B_T V T
$$

**Heat capacity (constant pressure)**

$$
G(T,P) = \min_V [E(V) + F_{vib}(V,T) + PV]
$$

**Gibbs free energy**

$$
S(T) = -\left(\frac{\partial F}{\partial T}\right)_V
$$

**Entropy**

$$
B_T(T) = -V\left(\frac{\partial P}{\partial V}\right)_T
$$

**Bulk modulus**

### Computational Workflow

```
1. Equation of State (EOS)
   └─ Relax structure at ~9 volumes
   └─ Fit E(V) → Get V₀, B₀

2. QHA Volume Grid
   └─ Generate 5-11 volumes around V₀
   └─ Typically ±5-10% range

3. Phonon Calculations
   └─ Calculate phonons at each volume
   └─ Each phonon = supercell force calculation

4. QHA Analysis
   └─ Construct G(V,T,P) = E(V) + F_vib(V,T) + PV
   └─ Find V_eq(T,P) = arg min G(V,T,P)
   └─ Extract α, Cv, Cp, S, B from V_eq(T,P)
```

---

## Configuration Options

### Example Types

The tutorial provides 5 comprehensive examples:

#### 1. Basic (Silicon)
```python
EXAMPLE_TYPE = "basic"
```
- Material: Silicon (diamond cubic)
- Temperature: 0-1000 K (11 points)
- Pressure: Ambient (0 GPa)
- Purpose: Learn basic QHA workflow
- Settings: Standard (2×2×2 k-points, 2×2×2 supercell)

#### 2. High Pressure (MgO)
```python
EXAMPLE_TYPE = "high_pressure"
```
- Material: MgO (rocksalt)
- Temperature: 300-2000 K (7 points)
- Pressure: 0-100 GPa (6 pressures)
- Purpose: Create P-T phase diagram, study Earth's mantle
- Settings: More volumes (7), Birch-Murnaghan EOS

#### 3. Custom Accuracy (Diamond)
```python
EXAMPLE_TYPE = "custom_accuracy"
```
- Material: Diamond (very hard)
- Temperature: 300-1200 K (4 points)
- Accuracy: Publication-quality (TZP, 500 Ry, 8×8×8 k-points)
- Purpose: High-accuracy calculations
- Settings: Large supercell (3×3×3), dense EOS (11 volumes)

#### 4. Metal (Aluminum)
```python
EXAMPLE_TYPE = "metal"
```
- Material: Aluminum (FCC)
- Temperature: 300-900 K (4 points)
- Special: Handles imaginary modes from electronic smearing
- Purpose: Demonstrate metal-specific settings
- Settings: MP occupation, ignore_imaginary_modes=True

#### 5. Wide Temperature (NaCl)
```python
EXAMPLE_TYPE = "wide_temperature"
```
- Material: NaCl (rocksalt)
- Temperature: 0-1200 K (21 points, non-uniform)
- Purpose: Comprehensive thermal characterization
- Settings: Dense sampling at low T (quantum regime)

---

## Quick Start

```bash
# 1. Preview QHA workflow structure
# Edit tutorial.py: EXAMPLE_TYPE = "basic", RUN_MODE = "dry_run"
python tutorial.py

# 2. Inspect workflow
ls preview_output/job_*
# Should see: EOS jobs + QHA phonon jobs

# 3. Try different examples
# Edit tutorial.py: EXAMPLE_TYPE = "high_pressure"
python tutorial.py

# 4. Run calculation (WARNING: 3-5 hours!)
# Edit tutorial.py: RUN_MODE = "local"
python tutorial.py
```

---

## Expected Output

### Dry-Run Mode

```
✅ Dry-run complete!

📁 QHA Workflow Structure:
  1. EOS Calculation: ~9 volume points
     └─ Each volume: Relaxation job
  2. QHA Phonon Calculations: 5-7 volumes
     └─ Each volume: Phonon calculation (supercell)
  3. QHA Analysis: Thermodynamic properties

💡 Total jobs: ~15-20 (depending on settings)
```

**Directory Structure**:
```
preview_output/
├── job_*_eos_001/        # EOS relaxation at V1
├── job_*_eos_002/        # EOS relaxation at V2
├── ...
├── job_*_phonon_001/     # QHA phonon at V1
├── job_*_phonon_002/     # QHA phonon at V2
└── ...
```

### Local Mode

```
✅ QHA calculation complete!

📊 Output includes:
  - Thermal expansion coefficient α(T)
  - Heat capacities Cv(T) and Cp(T)
  - Gibbs free energy G(T,P)
  - Entropy S(T)
  - Temperature-dependent bulk modulus B(T)
  - Equilibrium volume V(T)
```

---

## Analyzing Results

### Validation Checks

QHA results should satisfy:

✓ **Dulong-Petit Limit**: Cv → 3NkB at high T (24.94 J/(mol·K) per atom)
✓ **Third Law**: α → 0 as T → 0
✓ **Monotonicity**: V(T) increases for normal materials
✓ **Thermodynamic Requirement**: Cp > Cv always

### Comparison with Experiment

Typical QHA accuracy:
- **Thermal expansion α**: ±10-20% (better at low T)
- **Heat capacity Cv**: ±5-10% (very good)
- **Bulk modulus B**: ±5-15%

### Plotting Example

```python
import matplotlib.pyplot as plt
import numpy as np

# Load results (structure depends on output schema)
# temps, alpha, cv, cp, volume, bulk_mod = load_qha_results()

# Create publication-quality plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Thermal expansion
axes[0,0].plot(temps, alpha*1e6, 'b-', linewidth=2)
axes[0,0].set_xlabel('Temperature (K)', fontsize=12)
axes[0,0].set_ylabel(r'$\alpha$ (10$^{-6}$ K$^{-1}$)', fontsize=12)
axes[0,0].set_title('Thermal Expansion Coefficient')
axes[0,0].grid(True, alpha=0.3)

# Heat capacity
axes[0,1].plot(temps, cv, 'r-', linewidth=2, label='$C_V$')
axes[0,1].plot(temps, cp, 'b--', linewidth=2, label='$C_P$')
axes[0,1].axhline(y=24.94, color='k', linestyle=':', label='Dulong-Petit')
axes[0,1].set_xlabel('Temperature (K)', fontsize=12)
axes[0,1].set_ylabel('Heat Capacity (J/mol·K)', fontsize=12)
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Volume expansion
v0 = volume[0]
axes[1,0].plot(temps, (volume-v0)/v0*100, 'g-', linewidth=2)
axes[1,0].set_xlabel('Temperature (K)', fontsize=12)
axes[1,0].set_ylabel('Volume Change (%)', fontsize=12)
axes[1,0].set_title('Thermal Expansion')

# Bulk modulus
axes[1,1].plot(temps, bulk_mod, 'm-', linewidth=2)
axes[1,1].set_xlabel('Temperature (K)', fontsize=12)
axes[1,1].set_ylabel('Bulk Modulus (GPa)', fontsize=12)

plt.tight_layout()
plt.savefig('qha_properties.png', dpi=300)
```

---

## Convergence & Best Practices

### Number of Volumes

- **Minimum**: 3 (NOT recommended)
- **Standard**: 5-7 volumes (this tutorial)
- **High accuracy**: 9-11 volumes
- **Rule**: More volumes = better G(V,T) surface fitting

### Volume Range (`volume_factor`)

Controls how wide the volume range is:
- **Soft materials**: 0.90-0.92 (±8-10%)
- **Normal materials**: 0.93-0.95 (±5-7%) - default
- **Hard materials**: 0.95-0.97 (±3-5%)

Example: `volume_factor=0.92` → volumes from V₀×0.92 to V₀/0.92

### Temperature Grid

- **Low T** (0-200 K): Dense sampling (ΔT = 50 K) for quantum effects
- **Room T** (200-600 K): Standard sampling (ΔT = 100 K)
- **High T** (> 600 K): Coarse sampling (ΔT = 100-200 K)
- **Near transitions**: Very dense sampling

### EOS Choice

- **vinet**: Best general choice, handles compression well
- **birch_murnaghan**: Good for high pressure, ionic solids (this tutorial uses for MgO)
- **murnaghan**: Simple, less accurate at high compression

### Computational Cost

For Si (2 atoms), 2×2×2 supercell, basic settings:
- **EOS**: 9 relaxations (~2-5 min each) = 20-45 min
- **QHA**: 5 phonon calculations (~30-60 min each) = 2.5-5 hours
- **Total**: ~3-5 hours on modern workstation

Scales with:
- System size (supercell atoms)
- K-point density
- Basis set quality
- Number of volumes

### Critical Convergence Tests

Always test convergence of:
1. ✅ **Supercell size** for phonons (see tutorial 16)
2. ✅ **K-point mesh** density
3. ✅ **Number of volumes** in QHA (compare 5 vs 7 vs 9)
4. ✅ **Volume range** (volume_factor)
5. ✅ **EOS fit quality** (plot E vs V, check R²)

---

## QHA Limitations

### QHA is Valid When:

✅ Temperature not too high (T < 0.5-0.7 × T_melt)
✅ Material not strongly anharmonic
✅ Phonons remain stable (no imaginary modes, except metals)
✅ Single-phase system

### QHA Breaks Down When:

❌ **High Temperature** (T > 0.7 T_melt):
   - Intrinsic anharmonicity becomes important
   - Need full anharmonic methods (phono3py)

❌ **Phase Transitions**:
   - QHA assumes single phase
   - Cannot predict transition temperatures accurately
   - Discontinuities in G(T,P) not captured

❌ **Strongly Anharmonic Systems**:
   - Perovskites with soft modes
   - Ferroelectrics near Tc
   - Materials with low-lying optic modes
   - Large amplitude vibrations

❌ **Liquids and Melting**:
   - Harmonic approximation invalid
   - Need molecular dynamics or advanced methods

### When to Use QHA:

✅ Thermal expansion at moderate T (T < 0.6 T_melt)
✅ Heat capacity calculations
✅ Equation of state at high pressure
✅ Materials screening and high-throughput
✅ Bulk thermodynamic properties

### When NOT to Use QHA:

❌ Thermal conductivity (need phono3py for phonon-phonon scattering)
❌ High-T thermodynamics near melting
❌ Strongly anharmonic materials
❌ Dynamic properties (viscosity, diffusion)
❌ Surface properties (different thermal expansion)

---

## Common Issues

### Issue 1: Imaginary Frequencies

**Symptoms**:
```
Phonon calculation has negative frequencies
QHA analysis fails or gives unphysical results
```

**Solutions**:
1. **Ensure full relaxation**: Check forces < 0.02 eV/Å
2. **Check stability**: Material may be dynamically unstable
3. **For metals**: Use `ignore_imaginary_modes=True` (metal example)
4. **Increase supercell**: Larger supercell → better phonon sampling
5. **Check SCF**: Ensure tight SCF convergence (DM.Tolerance < 1e-5)

**Metal-Specific**:
Metals often have small imaginary modes near Γ due to electronic smearing. The `ignore_imaginary_modes=True` flag filters these artifacts.

### Issue 2: Unphysical Properties

**Symptoms**:
```
Negative thermal expansion
Heat capacity Cp < Cv
Heat capacity > Dulong-Petit limit
Volume decreases with temperature
```

**Solutions**:
1. **More volumes**: Increase from 5 to 7 or 9
2. **Check phonon convergence**: Supercell size, k-points
3. **Verify EOS fit**: Plot E vs V, check fit quality (R² > 0.999)
4. **Adjust volume range**: Ensure appropriate for material hardness

### Issue 3: Slow Convergence

**Symptoms**:
```
Properties change significantly with settings
Different number of volumes gives very different results
```

**Solutions**:
1. **Larger supercell**: Increase phonon supercell (2×2×2 → 3×3×3)
2. **Tighter SCF**: DM.Tolerance = 1e-6
3. **More QHA volumes**: Use 7 or 9 instead of 5
4. **Denser k-points**: Especially for metals
5. **Better basis**: DZP → TZP for high accuracy

### Issue 4: Long Calculation Time

**Solutions**:
1. **Start small**: Use 2×2×2 supercell for testing
2. **Reduce k-points**: Test with [2,2,2] first
3. **Fewer volumes**: Start with 5 volumes
4. **Use HPC**: Submit to cluster (RUN_MODE="submit")
5. **Parallel**: Use jobflow-remote for parallel execution

### Issue 5: EOS Fit Problems

**Symptoms**:
```
Poor fit quality (R² < 0.99)
Unphysical bulk modulus
Volume range doesn't bracket minimum
```

**Solutions**:
1. **More EOS points**: Increase from 9 to 11 volumes
2. **Better initial structure**: Ensure well-relaxed starting point
3. **Adjust volume range**: Use appropriate volume_factor
4. **Try different EOS**: vinet vs birch_murnaghan vs murnaghan
5. **Check relaxation**: Ensure all EOS points converged

---

## Advanced Topics

### Creating P-T Phase Diagrams

```python
# Calculate Gibbs free energies for two phases
# Phase with lower G(T,P) is stable

# Example: Compare two crystal structures
maker1 = SiestaQhaMaker(
    name="Phase 1",
    pressure=[0, 10, 20, 50, 100],  # GPa
    temperature=[300, 600, 900, 1200],  # K
)

maker2 = SiestaQhaMaker(
    name="Phase 2",
    pressure=[0, 10, 20, 50, 100],
    temperature=[300, 600, 900, 1200],
)

# Run both
flow1 = maker1.make(structure1)
flow2 = maker2.make(structure2)

# Post-process to find phase boundary where G1 = G2
```

### Combining with Elastic Constants

```python
# QHA gives:
# - B(T): Bulk modulus vs temperature
# - α(T): Thermal expansion

# Elastic constants give:
# - C11, C12, C44 at T=0

# Can study mechanical stability:
# - Born criteria vs temperature
# - Softening of elastic constants
```

### High-Throughput QHA

```python
# Screen multiple materials for thermal properties
from jobflow import Flow

materials = [si, mgo, diamond, nacl]  # List of structures
qha_jobs = []

for structure in materials:
    maker = SiestaQhaMaker(
        number_of_frames=5,
        temperature=[300, 600, 900],
        pressure=0.0
    )
    qha_jobs.append(maker.make(structure))

# Run all
workflow = Flow(qha_jobs)
results = run_locally(workflow, create_folders=True)
```

---

## Tips for Success

✅ **Always start with dry-run**: Verify workflow before 3-hour calculation
✅ **Test convergence systematically**: Supercell, k-points, volumes
✅ **Use basic settings first**: 2×2×2 supercell, [2,2,2] k-points for testing
✅ **For metals**: Always use `ignore_imaginary_modes=True`
✅ **Check EOS fit**: Plot E vs V, ensure R² > 0.999
✅ **Validate thermodynamics**: Cp > Cv, Cv → 3NkB, α → 0 as T → 0
✅ **Compare with experiment**: Benchmark α, Cv against literature
✅ **Document settings**: Keep notes on what works for each material type
✅ **Save results**: QHA calculations are expensive!

---

## Best Practices

**Workflow Design**:
1. **Start simple**: Basic example with minimal settings
2. **Test convergence**: Increase settings systematically
3. **Production run**: Use custom_accuracy settings
4. **Validate**: Compare with experiment or literature

**For Each Material Type**:
- **Metals**: MP occupation, ignore_imaginary_modes, dense k-points
- **Insulators**: Standard settings, can use coarser k-points
- **Hard materials**: Smaller volume range (volume_factor~0.95)
- **Soft materials**: Larger volume range (volume_factor~0.90)

**Resource Management**:
1. **Estimate time**: ~5-7 phonons × (supercell time)
2. **Check disk space**: Each phonon = many files
3. **Use HPC wisely**: Submit to cluster for production
4. **Clean up**: Remove intermediate files after completion

---

## References

### Theory

1. **QHA Formalism**:
   - Born, M. & Huang, K. (1954). *Dynamical Theory of Crystal Lattices*. Oxford.
   - Wallace, D.C. (1972). *Thermodynamics of Crystals*. Wiley.

2. **Implementation**:
   - Togo, A. & Tanaka, I. (2015). *Scr. Mater.* 108, 1-5. (phonopy)
   - Toher, C. et al. (2014). *Phys. Rev. B* 90, 174107. (AFLOW-QHA)

### Applications

3. **Thermal Expansion**:
   - Carrier, P. et al. (2007). *Phys. Rev. B* 76, 064116.

4. **High Pressure**:
   - Karki, B.B. et al. (2000). *Rev. Geophys.* 38, 495. (Earth's mantle)

---

## Next Steps

After completing this tutorial:

1. **Understand workflow**: Review dry-run output, job structure
2. **Test convergence**: Vary supercell, k-points, number of volumes
3. **Try all examples**: Understand differences between material types
4. **Validate results**: Compare with experimental thermal expansion
5. **Advanced**: Create P-T diagrams, combine with elastic constants
6. **High-throughput**: Screen multiple materials for thermal properties

---

*Back to [05-vibrational-properties](../README.md) | [Main Tutorial Index](../../README.md)*
