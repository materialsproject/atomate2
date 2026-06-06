# Tutorial: ASE-Based NEB with SIESTA

**Category**: 05-barriers
**Difficulty**: Advanced
**Time**: ~5 min (dry-run), ~3-8 hours (full NEB with relaxation)

---

## Overview

Nudged Elastic Band (NEB) calculations using ASE optimizers with SIESTA force calculator. This workflow provides flexible NEB optimization with climbing image support, endpoint relaxation, and comprehensive analysis.

This tutorial demonstrates **AseNebFlowMaker**, which uses Python-based ASE optimizers (FIRE, BFGS, LBFGS) instead of SIESTA's native Lua NEB.

---

## What You'll Learn

- ASE-based NEB with SIESTA calculator
- Automatic endpoint relaxation workflow
- Multiple optimizer options (FIRE, BFGS, Per-Image-BFGS, LBFGS)
- Climbing image NEB for accurate transition states
- Per-image optimization vs collective optimization
- NEB progress monitoring and analysis
- When to use ASE NEB vs SIESTA Lua NEB

---

## Prerequisites

- **Required**: [01-NebDirectFlowMaker](../01-NebDirectFlowMaker/) - Basic NEB concepts
- **Required**: Understanding of transition state theory
- **Recommended**: [01-convergence](../../01-convergence/) - Converged parameters critical for barriers
- **Recommended**: Familiarity with ASE (Atomic Simulation Environment)

---

## Key Concepts

### ASE NEB vs SIESTA Lua NEB

| Aspect | ASE NEB (This Tutorial) | SIESTA Lua NEB |
|--------|------------------------|----------------|
| **Optimizer** | FIRE, BFGS, LBFGS (Python) | Lua script (native) |
| **Climbing Image** | Full support | Limited |
| **Parallelization** | Per-image (BFGS/LBFGS) | Collective |
| **Flexibility** | High (custom optimizers) | SIESTA-native only |
| **Checkpoint** | Full state saving | Basic |
| **Progress monitoring** | Real-time log files | Post-calculation only |
| **Best for** | Complex paths, CI-NEB, debugging | Quick calculations, simple barriers |

### When to Use ASE NEB

✅ **Use ASE NEB when**:
- Need climbing image NEB for accurate transition state
- Want per-image parallelization (each image on separate core)
- Require custom optimization strategies
- Need detailed progress monitoring
- Want to resume interrupted calculations

❌ **Use Lua NEB when**:
- Simple barrier calculation
- Prefer SIESTA-native workflow
- Don't need advanced features

### ASE Optimizers

**FIRE** (Fast Inertial Relaxation Engine):
- Adaptive time step
- Fast for well-behaved paths
- Good default choice

**BFGS** (Broyden-Fletcher-Goldfarb-Shanno):
- Quasi-Newton method
- Builds approximate Hessian
- Excellent convergence for smooth paths

**Per-Image-BFGS**:
- Each image optimized independently
- Parallelizable across images
- Best for HPC clusters

**Per-Image-LBFGS** (Limited-memory BFGS):
- Lower memory than BFGS
- Suitable for large systems (> 100 atoms)

### Climbing Image NEB

Standard NEB finds minimum energy path (MEP). Climbing Image NEB (CI-NEB) ensures the highest energy image climbs to the exact saddle point.

**Algorithm**:
1. Run standard NEB to find approximate MEP
2. Identify highest energy image
3. Invert spring force component for that image
4. Image climbs uphill to exact transition state

**Benefits**:
- Accurate transition state geometry
- Precise barrier height
- Suitable for calculating pre-exponential factors

**Cost**: ~10-20% more iterations

---

## Workflow Structure

```
AseNebFlowMaker (with relax_endpoints=True)
├── Initial structure relaxation
│   └── Fixed-cell relaxation (tight forces < 0.01 eV/Å)
├── Final structure relaxation
│   └── Fixed-cell relaxation (tight forces < 0.01 eV/Å)
├── Image generation
│   └── Interpolation between relaxed endpoints
├── NEB optimization
│   ├── Image 0 (initial - fixed)
│   ├── Image 1 (optimized)
│   ├── Image 2 (optimized)
│   ├── ... (N-2 mobile images)
│   ├── Image N-1 (optimized)
│   └── Image N (final - fixed)
└── Analysis
    ├── Energy profile extraction
    ├── Barrier height calculation
    └── Transition state identification
```

---

## Quick Start

### Basic Example (Without Endpoint Relaxation)

```python
from atomate2.siesta.flows.neb import AseNebFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from pymatgen.core import Structure
from jobflow import run_locally

# Load initial and final structures (already relaxed!)
initial = Structure.from_file("initial_relaxed.xsf")
final = Structure.from_file("final_relaxed.xsf")

# Create ASE NEB workflow
flow = AseNebFlowMaker(
    number_of_images=5,
    optimizer="FIRE",              # ASE optimizer
    fmax=0.05,                     # Force convergence (eV/Å)
    climbing_image=True,           # Use CI-NEB
    spring_constant=1.0,           # Spring between images
    static_maker=StaticMaker(),    # SIESTA calculator
)

# Run
workflow = flow.make(initial_structure=initial, final_structure=final)
results = run_locally(workflow, create_folders=True)
```

### With Endpoint Relaxation (Recommended)

```python
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.powerups import update_user_siesta_settings

# Configure relaxation maker (tight tolerances!)
relax_maker = RelaxMaker.fixed_cell_relaxation()

# Create NEB workflow with endpoint relaxation
flow = AseNebFlowMaker(
    number_of_images=5,
    optimizer="PER_IMAGE_BFGS",  # Per-image optimization
    fmax=0.05,
    climbing_image=True,
    relax_endpoints=True,        # Relax before NEB
    relax_maker=relax_maker,     # Use custom relax maker
)

workflow = flow.make(initial_structure=initial, final_structure=final)

# Apply SIESTA parameters to ALL jobs (relax + NEB)
workflow = update_user_siesta_settings(
    workflow,
    {
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [2, 2, 2],
        "Mesh.Cutoff": "300 Ry",
    }
)

results = run_locally(workflow, create_folders=True)
```

---

## Configuration Options

### Number of Images

```python
# Simple hop (nearest neighbor)
number_of_images=5

# Complex path (multi-step)
number_of_images=9

# High accuracy (smooth MEP)
number_of_images=11
```

**Rule**: Transition state should have ≥2 images nearby

### Optimizer Selection

```python
# Fast and robust (default)
optimizer="FIRE"

# Excellent convergence for smooth paths
optimizer="BFGS"

# For HPC parallelization (each image on separate core)
optimizer="PER_IMAGE_BFGS"

# Large systems (> 100 atoms)
optimizer="PER_IMAGE_LBFGS"
```

### Force Convergence

```python
# Loose (fast testing)
fmax=0.10  # eV/Å

# Standard (recommended)
fmax=0.05  # eV/Å

# Tight (accurate barrier)
fmax=0.02  # eV/Å

# Very tight (transition state geometry)
fmax=0.01  # eV/Å
```

### Climbing Image

```python
# Standard NEB
climbing_image=False

# Climbing Image NEB (recommended for accurate TS)
climbing_image=True
```

### Spring Constant

```python
# Weak springs (may allow image collapse)
spring_constant=0.5

# Standard (recommended)
spring_constant=1.0

# Strong springs (stiffer path, may prevent convergence)
spring_constant=2.0
```

---

## Output

### File Structure

```
job_XXX_Initial_Relaxation/
├── siesta.out             # Relaxation log
├── siesta.XV              # Relaxed initial structure
└── ...

job_YYY_Final_Relaxation/
├── siesta.out             # Relaxation log
├── siesta.XV              # Relaxed final structure
└── ...

job_ZZZ_NEB_Optimization/
├── neb_progress.log       # Real-time progress (CHECK THIS!)
├── image_0/               # Initial endpoint (fixed)
├── image_1/               # Mobile image
├── image_2/               # Mobile image
├── ...
├── image_N/               # Final endpoint (fixed)
├── ase_neb_info.txt       # Final barrier heights
└── neb_energy_profile.png # Energy vs reaction coordinate
```

### Monitoring Progress

```bash
# Watch NEB progress in real-time
tail -f job_*_NEB_Optimization/neb_progress.log
```

**Example output**:
```
Step    Time        Energy (eV)    Max Force (eV/Å)    Images
----    ----        -----------    -----------------    ------
  0     0.0 s       -245.32        0.245                [0.00, 0.31, 0.45, 0.37, 0.00]
 10     45.2 s      -245.28        0.118                [0.00, 0.29, 0.42, 0.35, 0.00]
 20     92.1 s      -245.25        0.067                [0.00, 0.28, 0.41, 0.33, 0.00]
...
156    752.8 s      -245.32        0.018                [0.00, 0.31, 0.45, 0.37, 0.00]
CONVERGED: Max force 0.018 < 0.05 eV/Å
```

### Analysis Output

```python
# Read barrier heights
with open("job_*_NEB_Optimization/ase_neb_info.txt") as f:
    print(f.read())
```

**Output**:
```
ASE NEB Calculation Summary
============================

Convergence Status: CONVERGED
Final max force: 0.018 eV/Å
Total steps: 156

Energy Barrier:
  Forward barrier:  0.45 eV
  Reverse barrier:  0.52 eV
  Transition state at image 3

Reaction Coordinate:
  Image 0 (init):  -245.32 eV (0.00 eV relative)
  Image 1:         -245.01 eV (0.31 eV relative)
  Image 2:         -244.89 eV (0.43 eV relative)
  Image 3 (TS):    -244.87 eV (0.45 eV relative) ← TRANSITION STATE
  Image 4:         -244.95 eV (0.37 eV relative)
  Image 5 (final): -244.80 eV (0.52 eV relative)
```

---

## Best Practices

✅ **MUST relax endpoints**: Set `relax_endpoints=True` with tight force tolerance
✅ **Use climbing image**: For accurate transition state → `climbing_image=True`
✅ **Adequate images**: 5-7 for simple hops, 9-11 for complex paths
✅ **Monitor progress**: Check `neb_progress.log` regularly
✅ **Converged SIESTA params**: k-points and basis critical for accurate barriers
✅ **Start with FIRE**: Good default, switch to BFGS if needed

❌ **Don't skip endpoint relaxation**: Residual forces → wrong barriers
❌ **Don't use too few images**: May miss transition state
❌ **Don't over-tighten fmax**: 0.02 eV/Å sufficient, 0.01 eV/Å rarely needed
❌ **Don't use unconverged k-points**: Barriers very sensitive!

---

## Why Relax Endpoints?

**NEB requires endpoints at local minima**:
- Unrelaxed endpoints have residual forces
- Forces propagate to NEB images
- Result: Incorrect barriers, poor convergence

**Workflow demonstrates automatic relaxation**:
```python
relax_endpoints=True   # Workflow handles it automatically
relax_maker=RelaxMaker.fixed_cell_relaxation(
    user_params={"MD.MaxForceTol": "0.001 eV/Ang"}  # Very tight!
)
```

**When NOT to relax**:
- Endpoints already fully relaxed in previous calculation
- Testing/debugging with approximate structures
- Comparing multiple NEB paths with same endpoints (reuse relaxed structures)

---

## Troubleshooting

**Problem**: NEB not converging after many iterations

**Solution**:
1. Increase number of images: `number_of_images=9`
2. Loosen force tolerance: `fmax=0.10` initially, then tighten
3. Try different optimizer: FIRE → BFGS or vice versa
4. Check spring constant: Try `spring_constant=0.5` or `2.0`

---

**Problem**: Barrier height changes significantly with parameters

**Solution**:
1. **Most likely**: Unconverged k-points or basis
   - Run convergence tests first!
   - Barriers very sensitive to computational parameters
2. Check endpoint relaxation: Forces < 0.01 eV/Å
3. Increase number of images
4. Use tighter fmax: 0.02 eV/Å instead of 0.05

---

**Problem**: Images collapsing to same structure

**Solution**:
1. Increase spring constant: `spring_constant=2.0`
2. Verify initial ≠ final structures
3. Check interpolation worked correctly
4. May indicate no barrier (downhill path)

---

**Problem**: Transition state at endpoint (image 0 or N)

**Solution**:
1. Verify endpoint structures are correct
2. May have swapped initial ↔ final
3. Barrier may be in opposite direction
4. Check forces on relaxed endpoints

---

## Comparison with NebDirectFlowMaker

**Use AseNebFlowMaker (this tutorial) when**:
- Need climbing image NEB
- Want per-image parallelization
- Require detailed progress monitoring
- Need to resume interrupted calculations
- Debugging complex NEB issues

**Use NebDirectFlowMaker (Lua) when**:
- Simple barrier calculation
- Prefer SIESTA-native workflow
- Don't need advanced features
- Want slightly simpler setup

**Both give same results** when converged!

---

## Next Steps

After completing ASE NEB workflow:

1. **Compare optimizers**: Try FIRE, BFGS, PER_IMAGE_BFGS on same system
2. **Convergence testing**: Test number of images, force tolerance, k-points
3. **Climbing image**: Compare CI-NEB vs standard NEB for transition state geometry
4. **Phonons at TS**: [06-vibrational-properties](../../06-vibrational-properties/) - Calculate pre-exponential factor
5. **Production NEB**: Apply to real defect migration problems

---

## Related Tutorials

- [01-NebDirectFlowMaker](../01-NebDirectFlowMaker/) - SIESTA Lua-based NEB
- [Barriers Overview](../README.md) - All NEB tutorials
- [01-convergence](../../01-convergence/) - Parameter convergence (do this first!)

---

**📚 [Back to Barrier Calculations](../README.md)** | **📖 [All Tutorials](../../../README.md)**
