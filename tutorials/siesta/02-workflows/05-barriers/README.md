# Tutorial: Defect Migration Workflows (NEB)

**Category**: 02-workflows/05-defects
**Difficulty**: Advanced
**Time**: ~5 min (dry-run), ~2-8 hours (full NEB calculation)

---

## Overview

This tutorial demonstrates **defect migration** calculations using the Nudged Elastic Band (NEB) method. NEB finds minimum energy paths (MEPs) and transition states for atomic migration, diffusion, and reaction pathways. These calculations are essential for understanding ion conductivity, catalytic reactions, and defect dynamics.

**What NEB Calculations Tell You**:
- **Migration barrier (Eb)**: Energy required for defect to hop (eV)
- **Transition state geometry**: Saddle point structure on MEP
- **Reaction coordinate**: Complete energy profile from initial → final
- **Diffusion mechanism**: Pathway and intermediate states
- **Pre-exponential factor**: For Arrhenius diffusion D = D₀ exp(-Eb/kT)

**Note**: This section focuses on **migration barriers** (NEB). For **defect formation energies** (vacancies, interstitials, substitutions), future tutorials are planned.

---

## What You'll Learn

- Using `NebDirectFlowMaker` for SIESTA Lua-based NEB
- Using `AseNebFlowMaker` for ASE optimizer-based NEB
- Using `NebVacancyExchangeFlowMaker` for atom exchange barriers
- Image generation and interpolation methods
- Convergence criteria for transition states
- NEB checkpoint/resume capabilities
- Post-processing: barrier heights, reaction coordinates

---

## Prerequisites

- **Required tutorials**: [01-basics/01-RelaxMaker](../../01-basics/01-RelaxMaker/)
- **Required knowledge**:
  - Transition state theory
  - NEB method basics
  - Defect chemistry concepts
- **Recommended**: [01-convergence](../01-convergence/) - Converged parameters critical for accurate barriers
- **Structure files**: Located in [00-structures](../../00-structures/)

---

## Key Concepts

### Nudged Elastic Band Method

NEB finds minimum energy paths (MEPs) by:

1. **Create image chain**: Interpolate N images between initial and final states
2. **Add spring forces**: Connect images with fictitious springs (prevent collapse)
3. **Project forces**: Remove spring force parallel to path, real force perpendicular to path
4. **Optimize images**: Move perpendicular to path to find MEP
5. **Extract barrier**: Maximum energy along path = activation energy

**Energy Expression**:

$$
E_{\text{barrier}} = \max_i E_i - E_{\text{initial}}
$$

Where $E_i$ is energy of image $i$ along the path.

### NEB Variants

**Standard NEB**:
- All images move toward MEP simultaneously
- Good for simple barriers

**Climbing Image NEB (CI-NEB)**:
- Highest energy image climbs to exact saddle point
- More accurate transition state
- Slightly more expensive

**ASE vs SIESTA Lua NEB**:
| Aspect | ASE (Python) | SIESTA Lua (FLOS) |
|--------|--------------|-------------------|
| **Optimizer** | BFGS, FIRE (ASE) | Lua NEB script |
| **Flexibility** | High (custom optimizers) | SIESTA-native only |
| **Checkpoint** | Full state saving | Limited |
| **Best for** | Complex paths, CI-NEB | Quick calculations |

### Number of Images

**Guidelines**:
- **Simple hops** (nearest neighbor): 5-7 images
- **Complex paths** (multi-step): 9-13 images
- **Molecular rotations**: 7-11 images
- **Rule of thumb**: Barrier peak should have ≥2 images nearby

**Too few images**: Miss transition state, underestimate barrier
**Too many images**: Expensive, diminishing returns

### Convergence Criteria

For accurate barriers:

```python
user_params = {
    "MD.MaxForceTol": "0.02 eV/Ang",  # NEB force tolerance
    "MD.MaxCGDispl": "0.1 Bohr",      # Max displacement per step
    "MD.NumCGSteps": 500,             # Max NEB iterations
}
```

**Typical values**:
- Force tolerance: 0.02-0.05 eV/Å (looser than relaxation)
- Max iterations: 200-500 (path optimization slower)

---

## Tutorial Subdirectories

### [01-NebDirectFlowMaker](01-NebDirectFlowMaker/)
**Description**: SIESTA Lua-based NEB calculations
**Tutorial Files**:
- `01_li_diffusion.py` - Li ion diffusion in MgO (5 images)
- `02_li_diffusion_relax.py` - With endpoint relaxation
- `03_li_diffusion_remote.py` - HPC submission example
- `to-test/` - Additional examples (vacancy diffusion, O vacancy, etc.)

**Features**:
- SIESTA native Lua NEB (`neb.lua`)
- Simpler setup than ASE
- Automatic interpolation
- Summary files with barrier heights

---

## Quick Start

### Example 1: Basic Li Diffusion

```python
from pymatgen.core import Structure
from atomate2.siesta.flows.neb import NebDirectFlowMaker
from atomate2.siesta.jobs.core import LuaMaker
from jobflow import run_locally

# Load initial and final structures
initial = Structure.from_file("../../00-structures/mgo_li-initial.xsf")
final = Structure.from_file("../../00-structures/mgo_li-final.xsf")

# Create NEB workflow
maker = NebDirectFlowMaker(
    number_of_images=5,
    relax_endpoints=False,  # Endpoints already relaxed
    neb_maker=LuaMaker.neb(
        use_custodian=True,
        user_params={
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [2, 2, 2],
            "Mesh.Cutoff": "300 Ry",
            "MD.MaxForceTol": "0.02 eV/Ang",
        },
    ),
    dry_run=True
)

# Generate and run
flow = maker.make(initial_structure=initial, final_structure=final)
results = run_locally(flow, create_folders=True)
```

### Example 2: With Endpoint Relaxation

```python
from atomate2.siesta.jobs.core import RelaxMaker

# Relax endpoints before NEB (recommended!)
relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [4, 4, 4],  # Denser for endpoint accuracy
        "MD.MaxForceTol": "0.01 eV/Ang",  # Tight!
    }
)

maker = NebDirectFlowMaker(
    number_of_images=7,
    relax_endpoints=True,
    endpoint_relax_maker=relax_maker,
    dry_run=True
)
```

### Example 3: ASE-Based NEB

```python
from atomate2.siesta.flows.neb import AseNebFlowMaker

# Use ASE optimizers with SIESTA calculator
maker = AseNebFlowMaker(
    number_of_images=7,
    optimizer="BFGS",      # ASE optimizer
    fmax=0.05,             # Force convergence (eV/Å)
    climb=True,            # Climbing image NEB
    use_custodian=True,
    dry_run=True
)

flow = maker.make(initial_structure=initial, final_structure=final)
```

### Example 4: Vacancy Exchange

```python
from atomate2.siesta.flows.neb import NebVacancyExchangeFlowMaker

# Automatically generates initial/final from vacancy swap
structure_with_vacancy = Structure.from_file("mgo_vacancy.cif")

maker = NebVacancyExchangeFlowMaker(
    number_of_images=5,
    vacancy_index=24,      # Site of vacancy
    swap_index=25,         # Atom to swap with vacancy
    dry_run=True
)

flow = maker.make(structure=structure_with_vacancy)
```

---

## Run Modes

### 1. Dry-Run Mode

```bash
python 01_li_diffusion.py  # With dry_run=True
```

**Output**:
```
preview_output/
├── job_relax_initial_*/      # (if relax_endpoints=True)
├── job_relax_final_*/        # (if relax_endpoints=True)
├── job_neb_*/                # Main NEB calculation
└── job_neb_analysis_*/       # Barrier extraction and plotting
```

**Check images**:
```bash
grep "Number of images" preview_output/job_neb_*/siesta.fdf
# Should show: 5 images (or your chosen number)
```

### 2. Local Execution

```bash
# Edit: Set dry_run=False
python 01_li_diffusion.py
```

**What happens**:
1. (Optional) Relax initial and final structures
2. Generate interpolated images
3. Run NEB optimization
4. Extract barrier height
5. Create `neb_summary.txt` with full analysis

**Time**: 2-8 hours depending on:
- System size (atoms in supercell)
- Number of images (5-13 typical)
- Convergence criteria (tighter = longer)
- K-points and basis size

---

## Expected Output

### NEB Summary File

After calculation, check `job_neb_analysis_*/neb_summary.txt`:

```
========================================
NEB Calculation Summary
========================================

1. Convergence Status
   • Final iteration: 156
   • Max force on images: 0.018 eV/Å
   • Status: CONVERGED (forces < 0.02 eV/Å)

2. Energy Barrier
   • Forward barrier:  0.45 eV
   • Reverse barrier:  0.52 eV
   • Transition state at image 3

3. Reaction Coordinate
   Image    Energy (eV)    Rel. Energy (eV)    Forces (eV/Å)
   -------- -------------- ------------------- --------------
   0 (init)    -245.32         0.00              0.000
   1           -245.01         0.31              0.024
   2           -244.89         0.43              0.019
   3           -244.87         0.45 (TS)         0.018
   4           -244.95         0.37              0.021
   5 (final)   -244.80         0.52              0.000

4. Geometry at Transition State
   Li position: (2.45, 2.45, 2.45) Å
   Li-O distances: [2.12, 2.08, 2.15, 2.10] Å
```

### Analyzing Results

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Read NEB results
with open("job_neb_analysis_*/neb_results.json") as f:
    neb_data = json.load(f)

images = np.array(neb_data['image_numbers'])
energies = np.array(neb_data['energies'])  # eV
forces = np.array(neb_data['max_forces'])  # eV/Å

# Find transition state
ts_index = np.argmax(energies)
barrier_forward = energies[ts_index] - energies[0]
barrier_reverse = energies[ts_index] - energies[-1]

print(f"Forward barrier: {barrier_forward:.3f} eV")
print(f"Reverse barrier: {barrier_reverse:.3f} eV")
print(f"Transition state at image {ts_index}")

# Plot MEP
plt.figure(figsize=(8, 6))
plt.plot(images, energies - energies[0], 'o-', lw=2)
plt.xlabel("Image number")
plt.ylabel("Relative energy (eV)")
plt.title(f"NEB: Li diffusion (Eb = {barrier_forward:.2f} eV)")
plt.axhline(0, color='k', ls='--', alpha=0.3)
plt.grid(alpha=0.3)
plt.savefig("neb_mep.png", dpi=300)
```

### Diffusion Coefficient

From barrier height, estimate diffusion coefficient:

```python
# Arrhenius equation: D = D₀ exp(-Eb / kT)
import numpy as np

Eb = 0.45  # eV (from NEB)
kB = 8.617e-5  # eV/K
T = 300  # K (room temperature)

# Typical pre-exponential for ion diffusion: 10⁻⁴ to 10⁻² cm²/s
D0 = 1e-3  # cm²/s

D = D0 * np.exp(-Eb / (kB * T))
print(f"Diffusion coefficient at {T} K: {D:.2e} cm²/s")

# Compare with experiments if available
D_exp = 5e-12  # cm²/s (example)
print(f"Experimental: {D_exp:.2e} cm²/s")
print(f"Ratio calc/exp: {D/D_exp:.1f}")
```

---

## Common Issues

### Issue 1: "NEB not converging"

**Symptoms**: Exceeds max iterations, forces still high

**Causes & Solutions**:

1. **Images too far apart**:
   ```python
   # Increase number of images
   maker = NebDirectFlowMaker(number_of_images=9)  # Instead of 5
   ```

2. **Initial path poor**:
   ```python
   # Use better interpolation (linear vs IDPP)
   # For ASE NEB:
   maker = AseNebFlowMaker(
       interpolation_method="idpp",  # Image-dependent pair potential
   )
   ```

3. **Tolerance too tight**:
   ```python
   user_params = {
       "MD.MaxForceTol": "0.05 eV/Ang",  # Looser (was 0.02)
   }
   ```

4. **Spring constant issues**:
   ```python
   # For ASE NEB, adjust spring constant
   maker = AseNebFlowMaker(k=0.5)  # Default 1.0, try 0.5 or 2.0
   ```

### Issue 2: "Barrier height unrealistic"

**Symptoms**: Eb too high/low compared to experiments

**Causes & Solutions**:

1. **Unconverged parameters**:
   - Run [01-convergence](../01-convergence/) first
   - Barriers very sensitive to k-points and basis!
   ```python
   user_params = {
       "a2s_kpts": [6, 6, 6],    # Denser
       "Mesh.Cutoff": "400 Ry",  # Higher
   }
   ```

2. **Endpoints not relaxed**:
   ```python
   # ALWAYS relax endpoints tightly
   maker = NebDirectFlowMaker(
       relax_endpoints=True,
       endpoint_relax_maker=RelaxMaker.fixed_cell_relaxation(
           user_params={"MD.MaxForceTol": "0.01 eV/Ang"}
       )
   )
   ```

3. **GGA limitations**:
   - PBE underestimates barriers by ~0.1-0.3 eV for ionic systems
   - Consider DFT+U for transition metals (see [03-advanced-features/12-dftu](../../03-advanced-features/02-physics-features/03-magnetic/))

4. **Supercell size effects**:
   - Defect-defect interactions if cell too small
   - Use ≥10 Å separation between periodic images

### Issue 3: "Transition state at endpoint"

**Symptoms**: Maximum energy at image 0 or N

**Cause**: Poor initial/final structures or wrong reaction coordinate

**Solutions**:

1. **Verify endpoint structures**:
   ```bash
   # Check forces on endpoints
   grep "siesta: Atomic forces" job_relax_*_/siesta.out | tail -1
   # Should be < 0.02 eV/Å
   ```

2. **Wrong direction**:
   - Swap initial ↔ final
   - Check that structures differ only by intended atomic position

3. **Multiple barriers**:
   - Path may have multiple steps
   - Run separate NEB for each elementary step

### Issue 4: "Images collapsing"

**Symptoms**: All images converge to same structure

**Causes**:

1. **Spring forces too weak** (ASE NEB):
   ```python
   maker = AseNebFlowMaker(k=2.0)  # Increase spring constant
   ```

2. **Initial path degenerate**:
   - Verify initial ≠ final before NEB
   - Check interpolation worked correctly

---

## NEB Best Practices

### 1. Always Relax Endpoints First

```python
# ✅ CORRECT
maker = NebDirectFlowMaker(
    relax_endpoints=True,
    endpoint_relax_maker=RelaxMaker.fixed_cell_relaxation(
        user_params={"MD.MaxForceTol": "0.01 eV/Ang"}  # Tight!
    )
)

# ❌ WRONG - Using unrelaxed structures
maker = NebDirectFlowMaker(relax_endpoints=False)
```

**Why**: Endpoint forces propagate to images → incorrect barriers

### 2. Use Adequate Supercell

For defect migration:

```python
from pymatgen.core import Structure

structure = Structure.from_file("unit_cell.cif")

# Generate supercell with ≥10 Å separation
supercell = structure.make_supercell([3, 3, 3])

# Check dimensions
print(f"Supercell lattice: {supercell.lattice.abc}")
# Should be > (10, 10, 10) Å
```

### 3. Converge NEB Parameters

Test barrier convergence with:

```python
# Test 1: Number of images
for n_images in [5, 7, 9, 11]:
    # Barrier should converge within ~0.05 eV

# Test 2: Force tolerance
for ftol in [0.05, 0.03, 0.02, 0.01]:
    # Check if barrier changes < 0.02 eV

# Test 3: K-points (most critical!)
for kpts in [[2,2,2], [4,4,4], [6,6,6]]:
    # Barrier must converge < 0.1 eV
```

### 4. Checkpoint Long Calculations

For NEB on HPC:

```python
maker = AseNebFlowMaker(
    number_of_images=11,
    optimizer="BFGS",
    enable_checkpoints=True,  # Save state every N iterations
)

# Can resume if job times out
```

---

## Tips for Success

✅ **MUST relax endpoints tightly**: Forces < 0.01 eV/Å
✅ **Adequate images**: 5-7 for simple hops, 9-13 for complex paths
✅ **Converged parameters**: k-points, basis, cutoff (see [01-convergence](../01-convergence/))
✅ **Large supercell**: ≥10 Å separation between defect images
✅ **Reasonable tolerance**: 0.02-0.05 eV/Å (don't over-converge)
✅ **Verify initial ≠ final**: Check structures differ by intended coordinate
✅ **Use custodian**: Automatic error recovery (use_custodian=True)

---

## Advanced Features

### Climbing Image NEB

For accurate transition state geometry:

```python
from atomate2.siesta.flows.neb import AseNebFlowMaker

maker = AseNebFlowMaker(
    number_of_images=9,
    climb=True,  # Enable climbing image
    fmax=0.02,   # Tighter for TS geometry
)
```

**When to use**:
- Need exact transition state structure
- Calculating pre-exponential factors (vibrations at TS)
- Comparing with experiments on TS geometry

### Variable-Cell NEB

For reactions involving cell changes (rare):

```python
# Not standard in NEB workflows
# Consider using variable-cell relaxation on endpoints only
relax_maker = RelaxMaker.variable_cell_relaxation()
```

**Note**: Most defect migration occurs at fixed cell volume.

### Multi-Step Reactions

For complex paths (A → B → C):

```python
from jobflow import Flow

# Step 1: A → B
neb1 = NebDirectFlowMaker(number_of_images=7).make(structA, structB)

# Step 2: B → C
neb2 = NebDirectFlowMaker(number_of_images=7).make(structB, structC)

# Combine
workflow = Flow([neb1, neb2])
```

---

## Next Steps

After completing NEB tutorials:

1. **Phonons at TS**: [06-vibrational-properties](../06-vibrational-properties/) - Calculate pre-exponential factor
2. **Charged defects**: Combine with [03-advanced-features/05-charged-calculations](../../03-advanced-features/02-physics-features/05-charge/)
3. **Surface diffusion**: [03-surfaces-adsorption](../03-surfaces-and-adsorption/) - Adsorbate migration on surfaces

---

## References

- **NEB Method**: Henkelman & Jónsson, J. Chem. Phys. 113, 9901 (2000)
- **Climbing Image**: Henkelman et al., J. Chem. Phys. 113, 9978 (2000)
- **SIESTA NEB**: SIESTA Manual, Lua scripting chapter
- **ASE NEB**: ASE documentation, Transition State Tools
- **Defect Migration**: Van der Ven et al., Acc. Chem. Res. 42, 364 (2009)

---

*Back to [02-workflows](../README.md) | [Main Tutorial Index](../../README.md)*
