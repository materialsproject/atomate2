# Equation of State (EOS) Workflow Tutorial

**Category**: Advanced Workflows
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~30-60 min (full calculation)
**Prerequisites**: Completed convergence studies (tutorials/02-convergence/)

---

## Overview

This tutorial demonstrates how to calculate the **Equation of State (EOS)** for crystalline materials using atomate2siesta. The EOS describes the relationship between volume, pressure, and energy, providing fundamental information about a material's compressibility and stability.

### What You'll Learn

- How to run multi-volume EOS workflows
- Fitting energy-volume data to various EOS models
- Extracting bulk modulus and equilibrium volume
- Interpreting EOS fit quality and physical parameters
- Choosing appropriate strain ranges and number of points

### Key Concepts

**Equation of State**: Mathematical relationship E(V) describing how a material's energy changes with volume

**Bulk Modulus (B₀)**: Resistance to uniform compression (GPa)
B₀ = -V(∂P/∂V)ₜ

**Equilibrium Volume (V₀)**: Volume at which energy is minimized (zero pressure)

**EOS Models**: Different mathematical forms for fitting E-V data:
- **Birch-Murnaghan**: General-purpose, most commonly used
- **Murnaghan**: Simpler, good for metals
- **Vinet**: Accurate over wide pressure ranges
- **Pourier-Tarantola**: Alternative to Birch-Murnaghan

---

## Quick Start

### 1. Run the Tutorial

```bash
cd tutorials/03-advanced-workflows/01-eos
python 01_basic.py
```

**What it does**:
- 7-point EOS with standard settings
- Strain range: -5% to +5%
- Dry-run mode enabled (preview without calculations)
- Demonstrates complete EOS workflow

### 2. Customize for Your Material

Edit `01_basic.py` and modify:

```python
# Change structure
structure = Structure.from_file("your_material.cif")

# Adjust strain range and points
flow = SiestaEosFlowMaker(
    dry_run=False,           # Set to False for real calculations
    linear_strain=(-0.06, 0.06),  # -6% to +6%
    number_of_frames=9,       # More points for better fit
)
```

### 3. Select Execution Mode

```python
# For preview (no calculations)
dry_run=True

# For local execution
dry_run=False

# For HPC submission (see tutorials/04-infrastructure/02-job-submission/)
```

---

## Understanding the Workflow

### Workflow Structure

```
EOS Workflow
├── Step 1: Initial relaxation (find equilibrium geometry)
├── Step 2: Generate strained structures
│   ├── Compression volumes (e.g., -5%, -3%, -1%)
│   ├── Equilibrium volume (0%)
│   └── Expansion volumes (e.g., +1%, +3%, +5%)
├── Step 3: Relax each strained structure (fixed volume)
├── Step 4: Collect E-V data points
├── Step 5: Fit to EOS models
├── Step 6: Extract physical properties
└── Step 7: Generate plots and summaries
```

### What Happens at Each Step

**Step 1 - Initial Relaxation**:
- Fully relaxes initial structure (cell + atoms)
- Finds equilibrium geometry at zero pressure
- Provides reference structure for strain application

**Step 2 - Structure Generation**:
- Applies volumetric strains to equilibrium structure
- Maintains symmetry during scaling
- Generates N structures (5-11 typical)

**Step 3 - Energy Calculations**:
- Relaxes atoms at each fixed volume
- Calculates minimum energy for each volume
- Ensures consistent force tolerance across all volumes

**Step 4-6 - EOS Fitting**:
- Fits E(V) data to multiple EOS models
- Extracts B₀, V₀, E₀ for each model
- Compares fit quality across models

**Step 7 - Analysis**:
- Generates publication-quality E-V plots
- Writes comprehensive text summary
- Provides physical interpretation

---

## Example Configuration

The tutorial (`01_basic.py`) uses these settings:

**Configuration**:
- Strain range: -5% to +5%
- Number of points: 7
- Dry-run mode: Enabled (preview only)

**When to customize**:
- For production calculations: Set `dry_run=False`
- For higher accuracy: Increase points to 9-11
- For extended range: Use `(-0.07, 0.07)` or wider
- For tighter convergence: Add custom parameters

**Expected runtime** (when dry_run=False):
- Standard (DZP, 300 Ry, 4×4×4): ~30-45 minutes (local)
- Tight (DZP, 350 Ry, 6×6×6): ~60-90 minutes (local)

---

## Understanding EOS Models

### Birch-Murnaghan EOS (Recommended)

**Equation** (3rd order):
```
E(V) = E₀ + (9V₀B₀/16) × [(V₀/V)^(2/3) - 1]² × [6 - 4(V₀/V)^(2/3)]
```

**Strengths**:
- Based on finite strain theory
- Physically motivated
- Works for most materials
- Most commonly reported in literature

**Use for**: General-purpose EOS fitting

### Murnaghan EOS

**Equation**:
```
E(V) = E₀ + (B₀V/B₁) × [(V₀/V)^B₁/(B₁-1) + 1] - V₀B₀/(B₁-1)
```

**Strengths**:
- Simpler functional form
- Good for metals
- Fewer parameters than Birch-Murnaghan

**Limitations**:
- Less accurate at high compressions
- Assumes linear B(P) relationship

**Use for**: Metallic systems, quick estimates

### Vinet EOS

**Equation**:
```
E(V) = E₀ + (9B₀V₀/η²) × [1 - (1 - ηx) × exp(η(1-x))]
where x = (V/V₀)^(1/3), η = 3(B₁-1)/2
```

**Strengths**:
- Excellent for wide pressure ranges
- Good high-compression behavior
- Universal EOS form

**Use for**: High-pressure studies, planetary interiors

### Pourier-Tarantola EOS

**Equation**:
```
E(V) = E₀ + (B₀V₀/6) × [(V₀/V)² - 1]² × [3 + (V₀/V)² × (B₁-4)]
```

**Strengths**:
- Logarithmic strain-based
- Alternative to Birch-Murnaghan
- Similar accuracy

**Use for**: Comparison with Birch-Murnaghan, validation

---

## Output Files

### eos_summary.txt

**Contents**:
```
================================================================================
EQUATION OF STATE (EOS) FIT RESULTS
================================================================================

Raw Data Points:
Volume (Ų)           Energy (eV)
----------------------------------------
38.456789            -215.234567
40.123456            -215.678901
41.789012            -215.789012  ← Minimum energy (V₀)
...

EOS Fit Parameters:
--------------------------------------------------------------------------------

BIRCH_MURNAGHAN EOS:
  E₀ (equilibrium energy):  -215.789012 eV
  V₀ (equilibrium volume):   41.234567 Ų
  B₀ (bulk modulus):         98.7654 GPa
  B₁ (derivative):           4.123456
  RMS error:                 0.0012 eV

MURNAGHAN EOS:
  E₀:  -215.789015 eV
  V₀:   41.234789 Ų
  B₀:   98.8123 GPa
  ...
```

**How to interpret**:

1. **Check RMS error**: Should be < 0.01 eV for good fit
2. **Compare models**: V₀ and B₀ should be similar across models (±5%)
3. **Physical values**: Check B₀ against literature/expectations
4. **B₁ range**: Typically 3-5 for most materials

### eos_fit.png

**Plot contents**:
- Red circles: Calculated E-V data points
- Solid lines: EOS fit curves for each model
- Vertical line: Equilibrium volume (V₀)
- Legend: Fit parameters for each model

**Visual quality checks**:
- Data points should fall on fit curves
- Minimum should be well-defined (parabolic shape)
- No outlier points far from fit
- Multiple models should overlap near minimum

---

## Choosing Workflow Parameters

### Strain Range

**Guidelines**:

| Material Type | Recommended Range | Reason |
|--------------|-------------------|--------|
| Metals | ±5% to ±6% | Moderate compressibility |
| Semiconductors | ±5% to ±7% | Standard range |
| Ionic compounds | ±6% to ±8% | Higher compressibility |
| Hard materials (diamond, SiC) | ±4% to ±5% | Low compressibility |
| Soft materials (polymers) | ±8% to ±10% | High compressibility |

**Extending the range**:
```python
"linear_strain": (-0.08, 0.08),  # -8% to +8%
```

**When to extend**:
- Fit doesn't capture curvature
- Studying high-pressure behavior
- Unusual compressibility

### Number of Points

**Recommendations**:

| Points | Quality | Use Case | Runtime Multiplier |
|--------|---------|----------|-------------------|
| 5 | Minimal | Testing only | 1× (fastest) |
| 7 | Standard | Most calculations | 1.4× |
| 9 | Good | Publication quality | 1.8× |
| 11 | Excellent | High accuracy | 2.2× |
| 13+ | Excessive | Usually unnecessary | 2.6×+ |

**Rule of thumb**: 7-9 points sufficient for most materials

**More points needed when**:
- Asymmetric E-V curve
- Phase transitions possible
- High-accuracy required
- Comparing multiple EOS models

### SIESTA Parameters

**Minimum requirements**:
```python
"Mesh.Cutoff": "300 Ry"    # Well-converged
"kpts": [4, 4, 4]           # Sufficient for bulk
"PAO.BasisSize": "DZP"      # Double-zeta polarized
```

**For publication quality**:
```python
"Mesh.Cutoff": "350-400 Ry"  # Tight convergence
"kpts": [6, 6, 6] to [8, 8, 8]  # Dense grid
"PAO.BasisSize": "DZP" or "TZP"  # High-quality basis
"MD.MaxForceTol": "0.01 eV/Ang"  # Strict forces
```

**Critical**: Use the SAME parameters you converged in tutorials/02-convergence!

---

## Best Practices

### 1. Always Converge First

**Before running EOS**:
```bash
# Complete these tutorials first:
cd tutorials/02-convergence/01-kpoints-mesh-cutoff
python 04_2_combined_multi_criteria.py  # Multi-criteria convergence

cd tutorials/02-convergence/02-basis-parameters
python 01_basic.py  # Find optimal PAO.EnergyShift and PAO.SplitNorm
```

**Then apply converged parameters to EOS workflow**

### 2. Start with Dry-Run

```python
dry_run=True  # In SiestaEosFlowMaker
```

**Verify**:
- Number of structures generated (should be N+1)
- Volume range is reasonable
- Structure files look correct
- No errors in structure generation

### 3. Use Consistent Parameters

**DO**:
- Use same k-points for all volumes
- Use same basis set for all calculations
- Use same convergence criteria

**DON'T**:
- Change parameters between volumes
- Mix different functionals (XC.functional)
- Use different pseudopotentials

### 4. Check Fit Quality

**Good fit indicators**:
- RMS error < 0.01 eV
- All models give similar V₀ (within ±2%)
- All models give similar B₀ (within ±5%)
- B₁ in range 3-5 for most materials
- Smooth E-V curve with clear minimum

**Bad fit indicators**:
- RMS error > 0.1 eV
- Models disagree significantly
- B₁ < 2 or B₁ > 7 (unusual)
- E-V curve has noise or outliers

### 5. Validate Results

**Compare with literature**:
```
Material | Literature B₀ (GPa) | Your Result
Si       | 98-100              | ?
Al       | 72-76               | ?
NaCl     | 24-26               | ?
Diamond  | 440-450             | ?
```

**Typical bulk moduli**:
- Very hard: > 300 GPa (diamond, c-BN)
- Hard: 100-300 GPa (transition metals, SiC)
- Moderate: 50-100 GPa (Si, Al, semiconductors)
- Soft: < 50 GPa (alkali metals, ionic crystals)

---

## Common Issues and Solutions

### Issue 1: "Fit fails for some EOS models"

**Symptoms**:
- Some models show "exception" or "failed"
- Only 1-2 models work

**Causes**:
- Insufficient data points
- Poor quality data (not converged)
- Inappropriate strain range

**Solutions**:
```python
# Increase number of points
"number_of_frames": 9  # or 11

# Extend strain range
"linear_strain": (-0.07, 0.07)

# Check convergence parameters
```

**Note**: It's normal for 1-2 models to fail. Use the models that work.

### Issue 2: "Unrealistic B₀ values"

**Symptoms**:
- B₀ >> or << literature values
- Negative B₀ (unphysical)

**Causes**:
- Unconverged k-points or mesh cutoff
- Too few data points
- Structural instability

**Solutions**:
```python
# Step 1: Check convergence (critical!)
# Revisit tutorials/02-convergence/

# Step 2: Add more data points
"number_of_frames": 11

# Step 3: Tighten force tolerance
"MD.MaxForceTol": "0.01 eV/Ang"

# Step 4: Check if structure is stable
# Look for imaginary phonon modes
```

### Issue 3: "Different EOS models give very different results"

**Symptoms**:
- V₀ varies > 5% between models
- B₀ varies > 10% between models

**Causes**:
- Insufficient strain range
- Poor fit quality
- Asymmetric E-V curve

**Solutions**:
```python
# Extend strain range symmetrically
"linear_strain": (-0.08, 0.08)

# Add more points
"number_of_frames": 11

# Check for phase transitions in this volume range
```

**When to worry**: If Birch-Murnaghan and Vinet differ significantly, something is wrong.

### Issue 4: "E-V curve is not smooth"

**Symptoms**:
- Jagged E-V data points
- Points don't fall on smooth curve
- Large RMS error

**Causes**:
- Unconverged SCF
- Inconsistent k-point sampling
- Poor numerical precision

**Solutions**:
```python
# Tighten SCF tolerance
"DM.Tolerance": "1.0e-5"

# Use denser k-points
"kpts": [6, 6, 6]

# Increase mesh cutoff
"Mesh.Cutoff": "350 Ry"

# Enable SCF mixing adjustment
"SCF.Mixer.Weight": 0.1
```

### Issue 5: "Workflow takes too long"

**Symptoms**:
- EOS calculation runs for hours

**Expected runtimes** (7 points, Si, 4-core):
- Basic (SZ, 200 Ry, 3×3×3): ~10 min
- Standard (DZP, 300 Ry, 4×4×4): ~30 min
- Tight (DZP, 350 Ry, 6×6×6): ~60 min
- Very tight (TZP, 400 Ry, 8×8×8): ~120 min

**Solutions**:
```python
# For testing: Reduce number of points
number_of_frames=5  # Testing only

# For production: Submit to HPC cluster
# See tutorials/04-infrastructure/02-job-submission/

# Or use dry-run mode to test workflow
dry_run=True  # Preview without calculations
```

### Issue 6: "No output files generated"

**Symptoms**:
- No eos_summary.txt or eos_fit.png
- Workflow completes but no analysis files

**Causes**:
- matplotlib not installed
- Workflow failed before postprocessing
- File write permissions

**Solutions**:
```bash
# Install matplotlib
pip install matplotlib

# Check job folders for errors
ls -lrt job_*/

# Check for error messages in siesta.out files
grep -i "error" job_*/siesta.out
```

---

## Understanding Results

### Bulk Modulus (B₀)

**Physical meaning**:
- Resistance to uniform compression
- Higher B₀ = harder to compress
- Units: GPa (gigapascals)

**Formula**:
```
B₀ = -V (∂P/∂V)ₜ = V (∂²E/∂V²)ₜ
```

**Typical values**:
```
Material    B₀ (GPa)   Category
Diamond     440        Superhard
c-BN        369        Superhard
SiC         220        Hard
Fe          170        Hard
Al2O3       250        Hard
Si          98         Moderate
Al          76         Moderate
MgO         160        Moderate
Cu          140        Moderate
NaCl        24         Soft
Na          7          Very soft
```

**Using B₀**:
- Mechanical stability indicator
- Compare with experimental values
- Predict behavior under pressure
- Screen hard materials

### Equilibrium Volume (V₀)

**Physical meaning**:
- Volume at zero external pressure
- Corresponds to energy minimum
- Temperature-dependent (EOS at 0 K)

**Using V₀**:
- Reference for thermal expansion
- Compare with experiment (extrapolate to 0 K)
- Input for phonon calculations
- Basis for defect calculations

### Pressure Derivative (B₁)

**Physical meaning**:
```
B₁ = (∂B/∂P)ₜ
```
- How bulk modulus changes with pressure
- Dimensionless
- Typically 3-5 for most materials

**Significance**:
- B₁ ~ 4: "Normal" material
- B₁ < 3: Unusual softening under pressure
- B₁ > 6: Unusual hardening under pressure

---

## Advanced Customization

### Custom Initial Relaxation

```python
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.flows.eos import SiestaEosFlowMaker

# Create custom relaxation maker
custom_relax = RelaxMaker.variable_cell_relaxation({
    "PAO.BasisSize": "DZP",
    "Mesh.Cutoff": "350 Ry",
    "kpts": [6, 6, 6],
    "MD.MaxForceTol": "0.01 eV/Ang",
})

# Use in EOS workflow
eos_maker = SiestaEosFlowMaker(
    initial_relax_maker=custom_relax,
    linear_strain=(-0.06, 0.06),
    number_of_frames=9,
)
```

### Postprocessing Only

If you already have E-V data:

```python
from pymatgen.analysis.eos import EOS
import numpy as np

# Your E-V data
volumes = np.array([38.5, 40.0, 41.5, 43.0, 44.5])  # Ų
energies = np.array([-215.23, -215.68, -215.79, -215.70, -215.50])  # eV

# Fit EOS
eos = EOS(eos_name="birch_murnaghan")
eos_fit = eos.fit(volumes, energies)

print(f"V₀ = {eos_fit.v0:.3f} Ų")
print(f"E₀ = {eos_fit.e0:.6f} eV")
print(f"B₀ = {eos_fit.b0_GPa:.2f} GPa")

# Plot
eos_fit.plot()
```

### Comparing Multiple Materials

```python
# Run EOS for multiple structures
structures = {
    "Si": Structure.from_file("Si.cif"),
    "Ge": Structure.from_file("Ge.cif"),
    "GaAs": Structure.from_file("GaAs.cif"),
}

results = {}
for name, struct in structures.items():
    eos_maker = SiestaEosFlowMaker(name=f"EOS_{name}")
    flow = eos_maker.make(struct)
    results[name] = run_locally(flow, create_folders=True)

# Compare B₀ values
for name, result in results.items():
    print(f"{name}: B₀ = {result['B0']:.2f} GPa")
```

---

## Next Steps

### After Completing This Tutorial

1. **Validate your setup**:
   - Compare Si results with literature (B₀ ~ 98 GPa)
   - Check fit quality (RMS < 0.01 eV)

2. **Apply to your material**:
   - Use converged parameters from tutorials/02-convergence
   - Run basic example first
   - Validate against literature if available

3. **Use V₀ for further calculations**:
   - Phonon calculations (tutorials/05-vibrational-properties/01-phonons/)
   - Surface studies (tutorials/06-surfaces-and-adsorption/)
   - Defect calculations

4. **Advanced EOS workflows**:
   - EOS with basis convergence (tutorials/03-advanced-workflows/02-eos-basis-convergence/)
   - Temperature-dependent properties (tutorials/05-vibrational-properties/02-qha/)

---

## References

### Foundational Papers

1. **Birch-Murnaghan EOS**:
   - Birch, F. (1947). "Finite Elastic Strain of Cubic Crystals". *Physical Review*, 71(11), 809–824.

2. **Murnaghan EOS**:
   - Murnaghan, F. D. (1944). "The Compressibility of Media under Extreme Pressures". *PNAS*, 30(9), 244–247.

3. **Vinet EOS**:
   - Vinet, P., et al. (1989). "A universal equation of state for solids". *J. Phys.: Condens. Matter*, 1, 1941.

### Online Resources

- [Wikipedia: Birch–Murnaghan equation of state](https://en.wikipedia.org/wiki/Birch%E2%80%93Murnaghan_equation_of_state)
- [Wikipedia: Murnaghan equation of state](https://en.wikipedia.org/wiki/Murnaghan_equation_of_state)
- [Materials Project EOS Documentation](https://docs.materialsproject.org/)

---

## Summary

**What we covered**:
- ✅ EOS theory and physical meaning
- ✅ Running multi-volume workflows
- ✅ Fitting E-V data to EOS models
- ✅ Extracting bulk modulus and equilibrium volume
- ✅ Choosing appropriate parameters
- ✅ Validating and interpreting results
- ✅ Troubleshooting common issues

**Key takeaways**:
1. Always converge parameters first (tutorials/02-convergence)
2. 7-9 points with ±5-6% strain is standard
3. Compare multiple EOS models for reliability
4. Validate B₀ against literature
5. Use dry-run mode to preview workflows

**Ready for**: Production EOS calculations, basis-dependent EOS studies (next tutorial), thermal property calculations

---

*Tutorial created: 2024-10-22*
*Last updated: 2024-10-22*
