# Tutorial: Mechanical Properties Workflows

**Category**: 02-workflows/04-mechanical
**Difficulty**: Advanced
**Time**: ~5 min (dry-run), ~1-2 hours (full workflow)

---

## Overview

This tutorial demonstrates calculations of mechanical properties using elastic constant workflows. The elastic tensor describes how a material responds to applied stress, providing information about stiffness, stability, and anisotropy.

**What Elastic Calculations Tell You**:
- **Elastic constants (Cᵢⱼ)**: Stiffness matrix components
- **Bulk modulus (B)**: Resistance to uniform compression
- **Shear modulus (G)**: Resistance to shear deformation
- **Young's modulus (E)**: Stiffness in tension
- **Poisson's ratio (ν)**: Lateral strain response
- **Mechanical stability**: Positive eigenvalues → stable

---

## What You'll Learn

- Using `ElasticFlowMaker` for elastic tensor calculations
- Strain perturbation method
- Extracting mechanical properties from elastic tensor
- Analyzing elastic anisotropy
- Mechanical stability criteria

---

## Prerequisites

- **Required tutorials**: [01-basics/01-RelaxMaker](../../01-basics/01-RelaxMaker/)
- **Required knowledge**: Elasticity theory basics
- **Recommended**: [01-convergence](../01-convergence/) - Converged parameters
- **Required**: **Fully relaxed structure** at equilibrium

---

## Key Concepts

### Elastic Tensor

The elastic tensor relates stress σ to strain ε via Hooke's law:

$$
\sigma_i = \sum_j C_{ij} \varepsilon_j
$$

For a crystal with symmetry:
- **Cubic**: 3 independent constants (C₁₁, C₁₂, C₄₄)
- **Hexagonal**: 5 independent constants
- **Orthorhombic**: 9 independent constants
- **Triclinic**: 21 independent constants

### Strain Perturbation Method

1. **Start with relaxed structure** (critical!)
2. **Apply small strains**: Typically ±1% (δ = 0.01)
3. **Calculate stress response**: Run static calculations
4. **Extract elastic constants**: From σ vs ε relationship
5. **Compute derived properties**: B, G, E, ν

**Typical number of calculations**:
- Cubic: ~6-12 strains
- Hexagonal: ~10-16 strains
- Lower symmetry: 20-30 strains

### Derived Mechanical Properties

From elastic tensor C, compute:

**Voigt average** (upper bound):
- $B_V = \frac{1}{9}(C_{11} + C_{22} + C_{33}) + \frac{2}{9}(C_{12} + C_{13} + C_{23})$
- $G_V = \frac{1}{15}(C_{11} + C_{22} + C_{33} - C_{12} - C_{13} - C_{23}) + \frac{1}{5}(C_{44} + C_{55} + C_{66})$

**Reuss average** (lower bound):
- $B_R = 1/(S_{11} + S_{22} + S_{33} + 2(S_{12} + S_{13} + S_{23}))$
- $G_R = 15/(4(S_{11} + S_{22} + S_{33}) - 4(S_{12} + S_{13} + S_{23}) + 3(S_{44} + S_{55} + S_{66}))$

**Voigt-Reuss-Hill average** (recommended):
- $B = (B_V + B_R)/2$
- $G = (G_V + G_R)/2$

**Young's modulus and Poisson's ratio**:
- $E = \frac{9BG}{3B + G}$
- $\nu = \frac{3B - 2G}{2(3B + G)}$

---

## Tutorial Subdirectories

### [01-ElasticFlowMaker](01-ElasticFlowMaker/)
**Description**: Complete elastic tensor calculation workflow
**Features**:
- Automatic strain generation based on symmetry
- Stress calculations for each strain
- Elastic tensor fitting
- Mechanical properties extraction
- Stability analysis

---

## Quick Start

### Example 1: Basic Elastic Calculation

```python
from pymatgen.core import Structure
from atomate2.siesta.flows.elastic import ElasticFlowMaker
from jobflow import run_locally

# Load RELAXED structure (critical!)
structure = Structure.from_file("relaxed_structure.cif")

# Create elastic workflow
elastic_flow = ElasticFlowMaker(
    strain_magnitudes=[0.01],  # 1% strain
    relax_maker_kwargs={
        "user_params": {
            "PAO.BasisSize": "DZP",
            "kpts": [10, 10, 10],   # Dense k-points important
            "Mesh.Cutoff": "350 Ry",
        }
    },
    dry_run=True
)

# Generate and run
workflow = elastic_flow.make(structure)
results = run_locally(workflow, create_folders=True)
```

### Example 2: Multiple Strain Magnitudes

```python
# Test with multiple strain magnitudes to check linearity
elastic_flow = ElasticFlowMaker(
    strain_magnitudes=[0.005, 0.01, 0.015],  # 0.5%, 1%, 1.5%
    dry_run=True
)
```

### Example 3: With Relaxation First

```python
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import Flow

# Always relax to equilibrium first!
relax_maker = RelaxMaker.variable_cell_relaxation(
    user_params={
        "MD.MaxForceTol": "0.005 eV/Ang",    # Very tight!
        "MD.MaxStressTol": "0.01 GPa",       # Very tight!
    }
)

elastic_flow = ElasticFlowMaker(strain_magnitudes=[0.01])

# Chain: relax → elastic
relax_job = relax_maker.make(structure)
elastic_job = elastic_flow.make(structure, prev_dir=relax_job.output.dir_name)

workflow = Flow([relax_job, elastic_job])
results = run_locally(workflow, create_folders=True)
```

---

## Run Modes

### 1. Dry-Run Mode

```bash
python elastic_workflow.py  # With dry_run=True
```

**Output**:
```
preview_output/
├── job_equilibrium_*/          # Initial equilibrium calculation
├── job_strain_+1_xx_*/         # Strain in xx direction (+)
├── job_strain_-1_xx_*/         # Strain in xx direction (-)
├── job_strain_+1_yy_*/
├── job_strain_-1_yy_*/
├── ... (many strain jobs)
└── job_elastic_analysis_*/     # Analysis and property extraction
```

**Check symmetry-reduced strains**:
```bash
ls -1 preview_output/job_strain_*/ | wc -l
# Cubic: ~12 jobs, Hexagonal: ~16 jobs, etc.
```

### 2. Local Execution

```bash
# Edit: Set dry_run=False
python elastic_workflow.py
```

**Time**: 1-2 hours for cubic systems (depends on number of strains and system size)

---

## Expected Output

### Elastic Tensor

```
job_elastic_analysis_*/
├── elastic_tensor.json         # Full 6×6 elastic tensor
├── mechanical_properties.txt   # B, G, E, ν
├── compliance_tensor.json      # S = C⁻¹
└── stability_analysis.txt      # Mechanical stability check
```

### Analyzing Results

```python
import json
import numpy as np

# Read elastic tensor
with open("job_elastic_analysis_*/elastic_tensor.json") as f:
    elastic_data = json.load(f)

C = np.array(elastic_data['elastic_tensor_voigt'])  # 6×6 matrix

print("Elastic Tensor (GPa):")
print(C)

# For cubic system (Si example)
print(f"\nIndependent constants:")
print(f"C₁₁ = {C[0,0]:.1f} GPa")
print(f"C₁₂ = {C[0,1]:.1f} GPa")
print(f"C₄₄ = {C[3,3]:.1f} GPa")

# Read mechanical properties
with open("job_elastic_analysis_*/mechanical_properties.txt") as f:
    props = f.read()
print("\n" + props)
```

### Expected Properties for Si (Example)

```
Bulk modulus (B):    ~98 GPa
Shear modulus (G):   ~68 GPa
Young's modulus (E): ~160 GPa
Poisson's ratio (ν): ~0.22
```

---

## Common Issues

### Issue 1: "Elastic constants don't satisfy stability criteria"

**Symptoms**: Negative eigenvalues or failed stability check

**Causes & Solutions**:

1. **Structure not at equilibrium**:
   ```python
   # Relax with VERY tight criteria
   relax_maker = RelaxMaker.variable_cell_relaxation(
       user_params={
           "MD.MaxForceTol": "0.001 eV/Ang",  # 40× tighter than default
           "MD.MaxStressTol": "0.001 GPa",
       }
   )
   ```

2. **Check forces/stress before elastic calculation**:
   ```bash
   grep "siesta: Atomic forces" relaxed_structure/siesta.out | tail -1
   # Should be < 0.01 eV/Å

   grep "siesta: Stress tensor" relaxed_structure/siesta.out | tail -6
   # Should be < 0.1 GPa
   ```

3. **Material actually unstable**:
   - Check phonons for imaginary modes
   - Might need different structure/phase

### Issue 2: "Elastic constants vary with strain magnitude"

**Symptoms**: Different C values for 0.5%, 1%, 1.5% strains

**Cause**: Nonlinear effects (strain too large)

**Solutions**:
```python
# Use smaller strains
elastic_flow = ElasticFlowMaker(
    strain_magnitudes=[0.005]  # 0.5% instead of 1%
)

# Or check linearity
elastic_flow = ElasticFlowMaker(
    strain_magnitudes=[0.003, 0.005, 0.007, 0.01],  # Multiple values
    # Should give consistent C within ~1%
)
```

### Issue 3: "Unrealistic mechanical properties"

**Symptoms**: B or G negative, or wildly off from experiments

**Causes**:
1. **Unconverged parameters** - k-points, basis, cutoff
2. **Structure not relaxed**
3. **GGA functional limitations**

**Solutions**:
1. Run convergence tests first ([01-convergence](../01-convergence/))
2. Use very tight relaxation criteria
3. Accept ~10% error from GGA
4. Compare TRENDS, not absolute values

### Issue 4: "Elastic tensor not symmetric"

**Symptoms**: $C_{ij} \neq C_{ji}$

**Cause**: Numerical noise, insufficient precision

**Solutions**:
```python
# Tighter convergence
relax_maker_kwargs={
    "user_params": {
        "DM.Tolerance": "1e-7",    # Very tight
        "kpts": [12, 12, 12],      # Very dense
    }
}

# Average to enforce symmetry (post-processing)
C_sym = (C + C.T) / 2
```

---

## Mechanical Stability Criteria

### Cubic System

For mechanical stability:
- $C_{11} > 0$
- $C_{44} > 0$
- $C_{11} > |C_{12}|$
- $C_{11} + 2C_{12} > 0$

### Hexagonal System

- $C_{11} > |C_{12}|$
- $C_{33}(C_{11} + C_{12}) > 2C_{13}^2$
- $C_{44} > 0$
- $C_{66} > 0$

**Check automatically**: Workflow includes stability analysis in output.

---

## Elastic Anisotropy

### Zener Anisotropy Ratio

For cubic systems:

$$A = \frac{2C_{44}}{C_{11} - C_{12}}$$

- $A = 1$: Elastically isotropic
- $A \neq 1$: Anisotropic (most materials)

```python
A = 2 * C[3,3] / (C[0,0] - C[0,1])
print(f"Zener anisotropy: A = {A:.3f}")

if abs(A - 1) < 0.1:
    print("Nearly isotropic")
else:
    print(f"Anisotropic ({'softer' if A < 1 else 'stiffer'} in <100> vs <111>)")
```

### Universal Anisotropy Index

$$A^U = 5\frac{G_V}{G_R} + \frac{B_V}{B_R} - 6 \geq 0$$

- $A^U = 0$: Isotropic
- $A^U > 0$: Anisotropic

---

## Tips for Success

✅ **MUST start with fully relaxed structure**: Forces < 0.01 eV/Å, stress < 0.1 GPa
✅ **Use very tight convergence**: This is the most critical requirement
✅ **Dense k-points**: 10×10×10 minimum, 12×12×12 better
✅ **Small strains**: 1% standard, 0.5% if nonlinearity suspected
✅ **Check stability criteria**: Verify all conditions satisfied
✅ **Compare with experiments**: Validate B, G, E
✅ **Test strain magnitude independence**: Should get same C for different strains

---

## Best Practices

### 1. Multi-Stage Workflow

```python
from jobflow import Flow

# Stage 1: Rough relax
rough_relax = RelaxMaker.variable_cell_relaxation(
    user_params={"MD.MaxForceTol": "0.04 eV/Ang"}
)

# Stage 2: Tight relax
tight_relax = RelaxMaker.variable_cell_relaxation(
    user_params={
        "MD.MaxForceTol": "0.001 eV/Ang",
        "MD.MaxStressTol": "0.001 GPa",
    }
)

# Stage 3: Elastic
elastic_flow = ElasticFlowMaker(strain_magnitudes=[0.01])

# Chain them
job1 = rough_relax.make(structure)
job2 = tight_relax.make(structure, prev_dir=job1.output.dir_name)
job3 = elastic_flow.make(structure, prev_dir=job2.output.dir_name)

workflow = Flow([job1, job2, job3])
```

### 2. Verify Equilibrium

```python
def check_equilibrium(structure_file):
    """Verify structure is at equilibrium before elastic calculation."""
    output = SiestaOutput(structure_file)

    max_force = max(np.linalg.norm(f) for f in output.forces[-1])
    max_stress = np.abs(output.stress[-1]).max()

    print(f"Max force: {max_force:.4f} eV/Å")
    print(f"Max stress: {max_stress:.4f} GPa")

    if max_force > 0.01:
        print("WARNING: Forces too large! Relax more tightly.")
        return False
    if max_stress > 0.1:
        print("WARNING: Stress too large! Relax more tightly.")
        return False

    print("✓ Structure at equilibrium")
    return True
```

---

## Next Steps

After completing elastic calculations:

1. **Compare with EOS**: [02-equation-of-states](../02-equation-of-states/) - Bulk modulus should agree
2. **Thermal expansion**: Combine with [06-vibrational-properties/03-qha](../06-vibrational-properties/03-SiestaQhaFlowMaker/)
3. **Pressure effects**: How elastic constants change with pressure
4. **Anisotropic properties**: Sound velocity anisotropy, directional Young's modulus

---

## References

- **Elasticity Theory**: Nye "Physical Properties of Crystals"
- **Elastic Constants**: Wallace "Thermodynamics of Crystals"
- **Stability Criteria**: Born & Huang "Dynamical Theory of Crystal Lattices"
- **Computational Methods**: Wu et al. PRB 76, 054115 (2007)

---

*Back to [02-workflows](../README.md) | [Main Tutorial Index](../../README.md)*
