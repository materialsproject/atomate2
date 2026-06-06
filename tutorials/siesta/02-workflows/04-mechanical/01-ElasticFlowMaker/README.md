# Tutorial: Elastic Tensor Calculation

**Category**: 04-mechanical
**Difficulty**: Advanced
**Time**: ~5 min (dry-run), ~1-2 hours (full workflow)

---

## Overview

Complete elastic tensor calculation using strain perturbation method. Determines full elastic constants matrix (Cᵢⱼ) and derives mechanical properties including bulk modulus, shear modulus, Young's modulus, and Poisson's ratio.

This tutorial demonstrates the **ElasticFlowMaker** workflow for calculating mechanical properties from stress-strain relationships.

---

## What You'll Learn

- Elastic tensor calculation using strain perturbation method
- Extracting elastic constants (Cᵢⱼ) from stress-strain data
- Computing derived mechanical properties (B, G, E, ν)
- Mechanical stability criteria and anisotropy analysis
- Importance of equilibrium structure for elastic calculations
- Strain magnitude selection and linearity verification

---

## Prerequisites

- **Required**: [01-RelaxMaker](../../../01-basics/01-RelaxMaker/) completed
- **Critical**: **Fully relaxed structure at equilibrium** (forces < 0.01 eV/Å, stress < 0.1 GPa)
- **Recommended**: [01-convergence](../../01-convergence/) - Converged k-points and basis parameters
- **Recommended**: Basic elasticity theory knowledge

---

## Key Concepts

### Elastic Tensor

The elastic tensor relates stress (σ) to strain (ε) via Hooke's law:

$
\sigma_i = \sum_j C_{ij} \varepsilon_j
$

**Independent Constants by Symmetry**:
- **Cubic**: 3 independent (C₁₁, C₁₂, C₄₄)
- **Hexagonal**: 5 independent
- **Orthorhombic**: 9 independent
- **Triclinic**: 21 independent

### Strain Perturbation Method

**Workflow**:
1. Start with **fully relaxed structure** (CRITICAL!)
2. Apply small strains: ε = ±0.01 (±1%)
3. Calculate stress response for each strain
4. Extract elastic constants from σ vs ε
5. Compute derived properties

**Why Equilibrium Matters**:
- Non-equilibrium structure → residual forces → incorrect stress
- Small deviation → large errors in Cᵢⱼ
- **Requirement**: Forces < 0.01 eV/Å, Stress < 0.1 GPa

### Derived Mechanical Properties

From elastic tensor C:

**Bulk Modulus (B)**: Resistance to uniform compression
- Voigt-Reuss-Hill average: B = (B_V + B_R) / 2

**Shear Modulus (G)**: Resistance to shear deformation
- Voigt-Reuss-Hill average: G = (G_V + G_R) / 2

**Young's Modulus (E)**: Stiffness in tension
- E = 9BG / (3B + G)

**Poisson's Ratio (ν)**: Lateral strain response
- ν = (3B - 2G) / (2(3B + G))

---

## Workflow Structure

```
ElasticFlowMaker
├── Equilibrium calculation (optional)
├── Strain perturbations (symmetry-reduced)
│   ├── +1% strain in xx direction
│   ├── -1% strain in xx direction
│   ├── +1% strain in yy direction
│   ├── -1% strain in yy direction
│   ├── +1% strain in zz direction
│   ├── -1% strain in zz direction
│   ├── Shear strains (xy, xz, yz)
│   └── ... (total depends on symmetry)
└── Analysis
    ├── Elastic tensor fitting
    ├── Mechanical properties extraction
    └── Stability analysis
```

**Number of Calculations**:
- Cubic system: ~12 strain calculations
- Hexagonal: ~16 calculations
- Lower symmetry: 20-30 calculations

---

## Quick Start

### Basic Example

```python
from atomate2.siesta.flows.elastic import ElasticFlowMaker
from pymatgen.core import Structure
from jobflow import run_locally

# Load RELAXED structure (critical!)
structure = Structure.from_file("relaxed_structure.cif")

# Create elastic workflow
flow = ElasticFlowMaker(
    strain_magnitudes=[0.01],  # 1% strain
)

# Run
workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True)
```

---

## Configuration Options

### Strain Magnitudes

```python
# Standard (1% strain)
strain_magnitudes=[0.01]

# Small (0.5% - for nonlinear materials)
strain_magnitudes=[0.005]

# Check linearity (multiple values)
strain_magnitudes=[0.005, 0.01, 0.015]
# Should give consistent Cᵢⱼ within ~1%
```

### SIESTA Parameters

```python
from atomate2.siesta.powerups import update_user_siesta_settings

flow = ElasticFlowMaker()
workflow = flow.make(structure)

# Update parameters
workflow = update_user_siesta_settings(
    workflow,
    {
        "PAO.BasisSize": "DZP",       # High-quality basis
        "a2s_kpts": [10, 10, 10],     # Dense k-points (critical!)
        "Mesh.Cutoff": "350 Ry",      # Converged cutoff
        "DM.Tolerance": "1e-6",       # Tight convergence
    }
)

results = run_locally(workflow, create_folders=True)
```

### With Tier Preset

```python
from atomate2.siesta.sets.tiers import apply_tier_preset

flow = ElasticFlowMaker()
workflow = flow.make(structure)

# Apply preset (automatically sets optimal parameters)
workflow = apply_tier_preset(workflow, "relax_standard")
```

---

## Output

### File Structure

```
job_elastic_analysis_*/
├── elastic_tensor.json            # Full 6×6 elastic tensor (GPa)
├── compliance_tensor.json         # Compliance S = C⁻¹ (GPa⁻¹)
├── mechanical_properties.txt      # B, G, E, ν
├── stability_analysis.txt         # Mechanical stability criteria
└── anisotropy_metrics.txt         # Zener ratio, universal anisotropy
```

### Elastic Tensor Example (Silicon)

```python
import json
import numpy as np

# Read elastic tensor
with open("job_elastic_analysis_*/elastic_tensor.json") as f:
    data = json.load(f)

C = np.array(data['elastic_tensor_voigt'])  # 6×6 matrix in GPa

print("Elastic Tensor (GPa):")
print(C)

# For cubic system
print(f"\nIndependent constants:")
print(f"C₁₁ = {C[0,0]:.1f} GPa")
print(f"C₁₂ = {C[0,1]:.1f} GPa")
print(f"C₄₄ = {C[3,3]:.1f} GPa")
```

**Output**:
```
Elastic Tensor (GPa):
[[165.8  63.9  63.9   0.0   0.0   0.0]
 [ 63.9 165.8  63.9   0.0   0.0   0.0]
 [ 63.9  63.9 165.8   0.0   0.0   0.0]
 [  0.0   0.0   0.0  79.6   0.0   0.0]
 [  0.0   0.0   0.0   0.0  79.6   0.0]
 [  0.0   0.0   0.0   0.0   0.0  79.6]]

Independent constants:
C₁₁ = 165.8 GPa
C₁₂ = 63.9 GPa
C₄₄ = 79.6 GPa
```

### Mechanical Properties

```python
# Read mechanical properties
with open("job_elastic_analysis_*/mechanical_properties.txt") as f:
    print(f.read())
```

**Output**:
```
Mechanical Properties (Voigt-Reuss-Hill Averages):
- Bulk Modulus (B):      97.9 GPa
- Shear Modulus (G):     67.8 GPa
- Young's Modulus (E):  159.6 GPa
- Poisson's Ratio (ν):    0.218
- B/G Ratio:              1.44 (ductile if > 1.75)
```

---

## Best Practices

✅ **CRITICAL - Use fully relaxed structure**: Forces < 0.01 eV/Å, stress < 0.1 GPa
✅ **Dense k-point mesh**: 10×10×10 minimum, 12×12×12 better
✅ **Tight SCF convergence**: DM.Tolerance="1e-6" or tighter
✅ **Converged parameters**: Run convergence tests first
✅ **Small strains**: 1% standard, 0.5% for nonlinear materials
✅ **Check stability criteria**: Verify all conditions satisfied
✅ **Compare with experiments**: Validate B, G, E values

❌ **Don't use unrelaxed structure**: Will give completely wrong Cᵢⱼ
❌ **Don't skip convergence**: Unconverged k-points/basis → unreliable results
❌ **Don't use large strains**: > 2% may cause nonlinear effects
❌ **Don't ignore warnings**: Failed stability = unphysical structure

---

## Relaxation Before Elastic Calculation

### Two-Stage Workflow (Recommended)

```python
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.flows.elastic import ElasticFlowMaker
from jobflow import Flow

# Stage 1: Tight variable-cell relaxation
relax_maker = RelaxMaker.variable_cell_relaxation(
    user_params={
        "MD.MaxForceTol": "0.001 eV/Ang",  # Very tight (40× default)
        "MD.MaxStressTol": "0.001 GPa",    # Very tight
        "a2s_kpts": [10, 10, 10],
        "DM.Tolerance": "1e-6",
    }
)

# Stage 2: Elastic calculation
elastic_flow = ElasticFlowMaker(strain_magnitudes=[0.01])

# Chain them
relax_job = relax_maker.make(structure)
elastic_job = elastic_flow.make(structure, prev_dir=relax_job.output.dir_name)

workflow = Flow([relax_job, elastic_job])
results = run_locally(workflow, create_folders=True)
```

### Verify Equilibrium

Before elastic calculation, check:

```bash
# Check final forces
grep "siesta: Atomic forces" relaxed_job/siesta.out | tail -1

# Check final stress
grep "siesta: Stress tensor" relaxed_job/siesta.out | tail -6
```

Should show:
- Forces: < 0.01 eV/Å (all components)
- Stress: < 0.1 GPa (all components)

---

## Mechanical Stability Criteria

### Cubic System

For mechanical stability, must satisfy:
- C₁₁ > 0
- C₄₄ > 0
- C₁₁ > |C₁₂|
- C₁₁ + 2C₁₂ > 0

```python
# Check stability
def check_cubic_stability(C):
    c11, c12, c44 = C[0,0], C[0,1], C[3,3]

    checks = [
        (c11 > 0, "C₁₁ > 0"),
        (c44 > 0, "C₄₄ > 0"),
        (c11 > abs(c12), "C₁₁ > |C₁₂|"),
        (c11 + 2*c12 > 0, "C₁₁ + 2C₁₂ > 0"),
    ]

    for passed, criterion in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {criterion}")

    return all(passed for passed, _ in checks)

is_stable = check_cubic_stability(C)
print(f"\nMechanically stable: {is_stable}")
```

### Hexagonal System

Stability criteria:
- C₁₁ > |C₁₂|
- C₃₃(C₁₁ + C₁₂) > 2C₁₃²
- C₄₄ > 0
- C₆₆ > 0

---

## Elastic Anisotropy

### Zener Anisotropy Ratio (Cubic)

$
A = \frac{2C_{44}}{C_{11} - C_{12}}
$

```python
A = 2 * C[3,3] / (C[0,0] - C[0,1])
print(f"Zener anisotropy: A = {A:.3f}")

if abs(A - 1) < 0.1:
    print("Nearly isotropic")
else:
    direction = "softer in <100>" if A < 1 else "stiffer in <100>"
    print(f"Anisotropic ({direction})")
```

**Interpretation**:
- A = 1: Elastically isotropic
- A < 1: Softer in <100> direction
- A > 1: Stiffer in <100> direction

**Example**: Silicon A ≈ 1.56 (moderately anisotropic)

---

## Troubleshooting

**Problem**: Elastic constants don't satisfy stability criteria

**Solution**:
1. Verify structure is at equilibrium (forces < 0.01 eV/Å)
2. Re-relax with tighter criteria: `MD.MaxForceTol="0.001 eV/Ang"`
3. Check for imaginary phonon modes (structural instability)

---

**Problem**: Cᵢⱼ values vary with strain magnitude

**Solution**:
1. Use smaller strains: `strain_magnitudes=[0.005]`
2. Test linearity with multiple values: `[0.003, 0.005, 0.007, 0.01]`
3. Check for structural phase transitions at large strains

---

**Problem**: Elastic tensor not symmetric (Cᵢⱼ ≠ Cⱼᵢ)

**Solution**:
1. Increase k-point density: `[12, 12, 12]` or higher
2. Tighter SCF: `DM.Tolerance="1e-7"`
3. Post-process: Symmetrize C = (C + C.T) / 2

---

**Problem**: Bulk modulus differs from EOS value

**Solution**:
- Small difference (< 5%) is normal
- Large difference → check:
  - Structure equilibration (most common cause)
  - k-point convergence
  - Strain magnitude (should be same for both)

---

## Comparison with Experiments

### Silicon Example

```python
# Experimental values (room temperature)
exp_data = {
    'C11': 165.7,  # GPa
    'C12': 63.9,   # GPa
    'C44': 79.6,   # GPa
    'B': 97.9,     # GPa
}

# Compare
for prop, exp_val in exp_data.items():
    calc_val = # ... extract from calculations
    error = abs(calc_val - exp_val) / exp_val * 100
    print(f"{prop}: Calc={calc_val:.1f} GPa, Exp={exp_val:.1f} GPa, Error={error:.1f}%")
```

**Expected Accuracy**:
- GGA (PBE): Typically within ~5-10% for Cᵢⱼ
- Bulk modulus: ~5% accuracy
- Trends more reliable than absolute values

---

## Next Steps

After completing elastic calculations:

1. **Compare with EOS**: [02-equation-of-states](../../02-equation-of-states/) - Bulk modulus should agree within ~5%
2. **Thermal expansion**: [06-vibrational-properties/03-SiestaQhaFlowMaker](../../06-vibrational-properties/03-SiestaQhaFlowMaker/) - Temperature-dependent elastic constants
3. **Pressure effects**: How Cᵢⱼ changes under pressure
4. **Sound velocities**: Compute from elastic tensor and density

---

## Related Tutorials

- [02-equation-of-states](../../02-equation-of-states/) - Bulk modulus from EOS (should match)
- [01-convergence](../../01-convergence/) - Parameter convergence (do this first!)
- [Mechanical Properties Overview](../README.md) - All mechanical property tutorials

---

**📚 [Back to Mechanical Properties](../README.md)** | **📖 [All Tutorials](../../../README.md)**
