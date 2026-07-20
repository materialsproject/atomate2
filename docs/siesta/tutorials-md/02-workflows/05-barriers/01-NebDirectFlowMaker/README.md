# Tutorial: NEB Calculations with NebDirectFlowMaker

**Category**: 05-barriers
**Difficulty**: Advanced
**Time**: ~5 min (dry-run), ~60-120 min (full calculation)
**Prerequisites**: Completed convergence studies, familiarity with Lua scripts

---

## Overview

This tutorial demonstrates how to perform **Nudged Elastic Band (NEB)** calculations using atomate2siesta to find minimum energy paths (MEPs) and activation barriers for atomic transitions, diffusion processes, and chemical reactions.

### What You'll Learn

- Understanding the NEB method for finding MEPs
- Setting up NEB calculations with FLOS/Lua scripts
- Generating intermediate images between initial and final states
- Running NEB workflows in SIESTA
- Analyzing energy profiles and extracting activation barriers
- Identifying transition states

### Key Concepts

**Nudged Elastic Band (NEB)**: Chain-of-states method for finding the minimum energy path between two configurations

**Minimum Energy Path (MEP)**: Lowest energy pathway connecting reactant and product states

**Activation Barrier (E_barrier)**: Energy difference between transition state and initial state
```
E_barrier = E_TS - E_initial
```

**Transition State**: Highest energy point along the MEP (saddle point on potential energy surface)

**Reaction Energy**: Energy difference between final and initial states
```
E_reaction = E_final - E_initial
```

---

## Quick Start

### 1. Prerequisites

**FLOS Library** (required):
```bash
# Clone FLOS (Flexible Lua Optimiz ation Scripting)
git clone https://github.com/siesta-project/flos.git

# Set environment variable
export FLOS_PATH="$PWD/flos"
```

**Configure in ~/.atomate2.yaml**:
```yaml
FLOS_PATH: /path/to/flos
```

### 2. Tutorial Files

This directory contains 6 comprehensive examples demonstrating Li diffusion in MgO:

**Basic Examples**:
- **`01_li_diffusion.py`**: Basic Li diffusion with 5 images (no endpoint relaxation)
  - Demonstrates three methods to customize SIESTA parameters
  - Uses `NebDirectFlowMaker` with user_params, input_set_generator, and powerups

**Endpoint Relaxation Examples**:
- **`02_li_diffusion_relax_endpoint_both.py`**: Relax both initial and final endpoints
  - Shows how endpoint relaxation improves barrier accuracy

- **`03_li_diffusion_relax_endpoint_initial.py`**: Relax only initial endpoint
  - Useful when final state is already relaxed

- **`04_li_diffusion_relax_endpoint_final.py`**: Relax only final endpoint
  - Useful when initial state is already relaxed

- **`05_li_diffusion_relax.py`**: Advanced relaxation configuration
  - Custom relaxation parameters for endpoints

**Remote Execution**:
- **`06_li_diffusion_remote.py`**: HPC cluster submission with jobflow-remote
  - Shows how to submit NEB calculations to remote clusters

### 3. NebDirectFlowMaker Usage

The `NebDirectFlowMaker` is used when you have initial and final structures ready:

```python
from atomate2.siesta.flows.neb import NebDirectFlowMaker
from pymatgen.core import Structure

# Load initial and final structures
initial = Structure.from_file("../../../00-structures/mgo_li-initial.xsf")
final = Structure.from_file("../../../00-structures/mgo_li-final.xsf")

# Create NEB workflow
maker = NebDirectFlowMaker(
    number_of_images=5,        # Intermediate images
    relax_endpoints=True       # Optionally relax initial/final
)
flow = maker.make(initial_structure=initial, final_structure=final)
```

### 4. Select Execution Mode

All tutorials support three execution modes:

```python
# Dry-run mode (preview only)
flow = NebDirectFlowMaker(number_of_images=5, dry_run=True)

# Local execution
flow = NebDirectFlowMaker(number_of_images=5)
results = run_locally(flow, create_folders=True)

# Remote execution (HPC)
# See 06_li_diffusion_remote.py for full example
```

### 5. Run Tutorial

```bash
cd tutorials/02-workflows/05-barriers/01-NebDirectFlowMaker
python 01_li_diffusion.py  # Basic example
```

---

## Understanding the NEB Method

### The NEB Algorithm

The NEB method finds the MEP by:

1. **Initial Setup**:
   - Define initial state (reactant)
   - Define final state (product)
   - Generate N intermediate "images" by interpolation

2. **Force Decomposition**:
   - Each image feels forces from SIESTA calculation
   - Forces decomposed into perpendicular and parallel components
   - Perpendicular force minimizes energy
   - Parallel force (spring) maintains even spacing

3. **Optimization**:
   - Images relax perpendicular to path
   - Springs prevent image collapse
   - Iterate until forces converge

4. **Result**:
   - Converged path is the MEP
   - Highest energy image ≈ transition state
   - Energy profile gives activation barrier

### Mathematical Formulation

**NEB Force on Image i**:
```
F_i^NEB = F_i^⊥ + F_i^||

Where:
F_i^⊥ = F_i^SIESTA - (F_i^SIESTA · τ̂_i) τ̂_i     (perpendicular component)
F_i^|| = k [(r_{i+1} - r_i) - (r_i - r_{i-1})]   (spring force)

τ̂_i = tangent vector to path at image i
k = spring constant
```

**Purpose of Each Component**:
- **F^⊥**: Minimizes energy perpendicular to path
- **F^||**: Keeps images evenly distributed
- **Result**: Smooth, evenly-spaced MEP

### Climbing Image NEB (CI-NEB)

**Improvement over standard NEB**:
- After initial convergence, highest energy image "climbs"
- This image maximizes energy along path
- Gives more accurate transition state geometry
- No spring force on climbing image

**Enable in SIESTA**:
```python
"NEB.Climbing": "T"
```

---

## Workflow Structure

### NEB Workflow Steps

```
NEB Workflow
├── Step 1: Define initial structure (reactant)
├── Step 2: Define final structure (product)
├── Step 3: Generate intermediate images
│   ├── Linear interpolation (IDPP available)
│   └── Create N images between endpoints
├── Step 4: Initialize NEB calculation
│   ├── Load all images
│   ├── Set spring constants
│   └── Configure convergence criteria
├── Step 5: NEB iterations
│   ├── Calculate forces on all images (SIESTA)
│   ├── Decompose into ⊥ and || components
│   ├── Apply spring forces
│   ├── Update image positions
│   └── Check convergence
├── Step 6: Extract results
│   ├── Energy profile E(s) vs reaction coordinate s
│   ├── Identify transition state
│   └── Calculate activation barrier
└── Step 7: Optional CI-NEB refinement
```

### Image Interpolation Methods

**Linear Interpolation** (default):
```python
# Simple linear interpolation between endpoints
r_i = r_initial + (i/N) × (r_final - r_initial)
```

**IDPP (Image Dependent Pair Potential)**:
- Better initial guess for complex paths
- Minimizes change in interatomic distances
- Recommended for large structural changes
- Available in pymatgen:

```python
from pymatgen.analysis.path_finder import NEBPathfinder
pathfinder = NEBPathfinder(initial_structure, final_structure)
images = pathfinder.interpolate(nimages=5, autosort_tol=0)  # IDPP
```

---

## Example System: Li Diffusion in MgO

All tutorials in this directory demonstrate **lithium (Li) diffusion in MgO** (magnesium oxide):

### Physical System

**Configuration**:
- System: Li interstitial in MgO lattice
- Crystal structure: Rock salt (face-centered cubic)
- Diffusion mechanism: Li ion hopping between interstitial sites
- Number of images: 5 intermediate + 2 endpoints = 7 total (default)
- Spring constant: 0.1 eV/Å² (standard)

**Parameters** (01_li_diffusion.py):
- Basis: DZP
- Mesh cutoff: 300 Ry
- K-points: 2×2×2
- Force tolerance: 0.02 eV/Å

**When to use**:
- Ion diffusion studies in ionic crystals
- Battery materials (Li transport)
- Solid electrolyte conductivity
- Learning NEB methodology

**Expected runtime**: ~30-60 minutes (local with 5 images)

**Physical interpretation**:
- Li+ ion hops between interstitial sites in MgO
- Activation barrier typically 0.5-2.0 eV for Li in oxides
- Controls ionic conductivity and battery performance

### Variations in Tutorials

**Basic** (`01_li_diffusion.py`):
- 5 images, no endpoint relaxation
- Standard parameters
- ~30 minutes runtime

**With Endpoint Relaxation** (`02-05`):
- Relax endpoints before/during NEB
- More accurate barriers
- ~45-60 minutes runtime

**Remote Execution** (`06_li_diffusion_remote.py`):
- HPC cluster submission
- Parallel execution
- ~10-20 minutes on cluster

---

## Understanding NEB Parameters

### Number of Images

**Guidelines**:

| Images | Total Points | Use Case | Runtime Factor |
|--------|--------------|----------|----------------|
| 3 | 5 | Testing only | 1× |
| 5 | 7 | Standard | 1.4× |
| 7 | 9 | Detailed | 1.8× |
| 9 | 11 | High accuracy | 2.2× |
| 11+ | 13+ | Complex paths | 2.6×+ |

**Total points** = number_of_images + 2 (endpoints)

**How to choose**:
- Simple barrier (single TS): 5-7 images
- Complex path (multiple TSs): 9-11 images
- Tight energy variations: more images
- Smooth landscape: fewer images

**Rule of thumb**: Start with 5, increase if path is not smooth

### Spring Constant (k)

**Typical values**: 0.05 - 0.2 eV/Å²

| Spring (eV/Å²) | Effect | Use When |
|----------------|--------|----------|
| 0.05 | Weak | Images naturally well-spaced |
| 0.10 | Standard | Most calculations |
| 0.15 | Strong | Images tend to bunch up |
| 0.20 | Very strong | Difficult convergence |

**Signs you need stronger spring**:
- Images collapse to endpoints
- Uneven spacing along path
- Gaps in MEP coverage

**Signs spring is too strong**:
- Slow convergence
- Oscillations in energy profile
- Images pulled away from true MEP

### Force Tolerance

**Typical values**:
```python
"MD.MaxForceTol": "0.02 eV/Ang"   # Standard
"MD.MaxForceTol": "0.01 eV/Ang"   # Tight (publication)
"MD.MaxForceTol": "0.05 eV/Ang"   # Loose (testing)
```

**Recommendations**:
- Activation barriers: 0.01-0.02 eV/Å
- Qualitative paths: 0.05 eV/Å
- Publication quality: 0.01 eV/Å or tighter

**Impact on barrier**:
- Looser tolerance: ±0.05-0.1 eV uncertainty
- Tight tolerance: ±0.01-0.02 eV uncertainty

---

## Output Analysis

### Energy Profile

**Generated file**: `neb_profile.png`

**X-axis**: Reaction coordinate (image index or arc length)
**Y-axis**: Energy relative to initial state (eV)

**What to look for**:
- ✅ Smooth curve connecting endpoints
- ✅ Clear maximum (transition state)
- ✅ Symmetric if reaction is symmetric
- ❌ Jagged profile (unconverged or too few images)
- ❌ Kinks or discontinuities (problematic)

**Example profile**:
```
E (eV)
  |          TS
  |          *
  |         / \
  |        /   \
  |       /     \
  |======/       \=====  ← E_barrier
  |   initial    final
  |__________________ Reaction coordinate
     0   1   2   3   4
```

### Extracting Key Quantities

**Activation Barrier (E_barrier)**:
```python
E_barrier = max(E_images) - E_images[0]
```

**Transition State Index**:
```python
TS_index = argmax(E_images)
```

**Reaction Energy (E_reaction)**:
```python
E_reaction = E_images[-1] - E_images[0]
```

**Reverse Barrier**:
```python
E_reverse = max(E_images) - E_images[-1]
```

### Interpreting Results

**Activation Barrier Magnitude**:

| E_barrier | Process Rate at 300 K | Time Scale |
|-----------|----------------------|------------|
| > 2.0 eV | Very slow | Hours to years |
| 1.0-2.0 eV | Moderate | Seconds to hours |
| 0.5-1.0 eV | Fast | Microseconds to seconds |
| < 0.5 eV | Very fast | Nanoseconds |

**Rate estimation** (Arrhenius):
```
k = ν₀ exp(-E_barrier / kT)

Where:
ν₀ ≈ 10¹³ s⁻¹ (attempt frequency)
k = Boltzmann constant = 8.617×10⁻⁵ eV/K
T = temperature (K)
```

**Example** (E_barrier = 1.0 eV, T = 300 K):
```
k ≈ 10¹³ × exp(-1.0 / (8.617×10⁻⁵ × 300))
  ≈ 10¹³ × exp(-38.7)
  ≈ 2.3 × 10⁻⁴ s⁻¹

τ = 1/k ≈ 4300 s ≈ 1.2 hours
```

### Force Convergence

**Check final forces**:
```bash
# Extract forces from each image
grep "max" image_*/siesta.out

# Should be < MD.MaxForceTol for all images
```

**Typical convergence**:
- Initial images: 0.1-0.5 eV/Å
- After 10-20 NEB iterations: < 0.05 eV/Å
- Well-converged: < 0.02 eV/Å

---

## Best Practices

### 1. Relax Endpoints First

```python
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import Flow

# Relax initial structure
relax_init = RelaxMaker.fixed_cell_relaxation().make(initial_structure)

# Relax final structure
relax_final = RelaxMaker.fixed_cell_relaxation().make(final_structure)

# Run relaxations
flow = Flow([relax_init, relax_final])
results = run_locally(flow, create_folders=True)

# Use relaxed structures for NEB
```

**Why**: Endpoints must be local minima for valid MEP

### 2. Use Consistent Parameters

**DO**:
- Same k-points for all images
- Same basis set across path
- Same XC functional
- Same pseudopotentials

**DON'T**:
- Change parameters between images
- Mix different levels of theory
- Use different convergence criteria

### 3. Start with Fewer Images

**Workflow**:
1. Run with 3-5 images to get rough path
2. Check if barrier is smooth
3. Add more images if needed (7-9)
4. Refine with CI-NEB for TS

**Saves time**: Cheaper to add images than start with many

### 4. Validate Initial Guess

**Use dry-run mode**:
```python
MODE = "dry_run"
```

**Check**:
- Interpolation looks reasonable
- No atomic overlaps in intermediate images
- Symmetry preserved if expected
- Structures are chemically sensible

### 5. Converge Parameters First

**Before NEB**:
```bash
# Ensure converged for your system
cd tutorials/02-convergence/01-kpoints-mesh-cutoff
python tutorial.py

cd tutorials/02-convergence/02-basis-parameters
python tutorial.py
```

**Then use same parameters for NEB**

### 6. Monitor Convergence

**During NEB**:
- Check forces are decreasing
- Energy profile becomes smoother
- Image spacing remains reasonable
- No images collapsing

**Typical signs of problems**:
- Forces not decreasing after 20 iterations
- Images bunching at endpoints
- Erratic energy fluctuations
- SCF convergence failures

---

## Common Issues and Solutions

### Issue 1: "FLOS library not found"

**Symptoms**:
```
Error: Lua script neb.lua not found
FLOS_PATH not set
```

**Solutions**:
```bash
# Install FLOS
git clone https://github.com/siesta-project/flos.git

# Set in ~/.atomate2.yaml
FLOS_PATH: /full/path/to/flos

# Or export in shell
export FLOS_PATH="/full/path/to/flos"

# Verify
ls $FLOS_PATH/flos/optim/neb.lua
```

### Issue 2: "Images collapse to endpoints"

**Symptoms**:
- All intermediate images move to initial or final state
- Energy profile shows no barrier
- Unphysical path

**Causes**:
- Spring constant too weak
- Initial path far from MEP
- Endpoints not at local minima

**Solutions**:
```python
# Increase spring constant
"NEB.Spring": 0.15  # or 0.20

# Relax endpoints first
# Ensure they are local minima

# Try IDPP interpolation
from pymatgen.analysis.path_finder import NEBPathfinder
pathfinder = NEBPathfinder(initial, final)
images = pathfinder.interpolate(nimages=5, autosort_tol=0)
```

### Issue 3: "NEB not converging"

**Symptoms**:
- Forces remain high after many iterations
- Energy profile oscillates
- SCF failures

**Solutions**:
```python
# Loosen force tolerance temporarily
"MD.MaxForceTol": "0.05 eV/Ang"

# Increase SCF iterations
"MaxSCFIterations": 600

# Adjust SCF mixing
"SCF.Mixer.Weight": 0.05  # Slower, more stable

# Reduce number of images
number_of_images = 3  # Start simple

# Check if endpoints are minima
# Run relaxation on initial/final separately
```

### Issue 4: "Barrier height seems wrong"

**Symptoms**:
- E_barrier doesn't match literature
- Value unrealistic (e.g., > 5 eV for simple hop)

**Causes**:
- Unconverged parameters (k-points, cutoff)
- Wrong pathway (not the MEP)
- Endpoints not relaxed
- XC functional issue

**Solutions**:
```python
# Check convergence (CRITICAL!)
# Revisit tutorials/02-convergence/

# Verify correct pathway
# Check literature for mechanism

# Tighten parameters
"Mesh.Cutoff": "350 Ry"
"kpts": [4, 4, 4]  # Or denser

# Use hybrid functionals if needed (expensive)
"XC.functional": "HSE06"  # For accurate barriers
```

### Issue 5: "Jagged energy profile"

**Symptoms**:
- E vs reaction coordinate is not smooth
- Large energy jumps between adjacent images

**Causes**:
- Unconverged SCF
- Inconsistent numerical precision
- Too few images

**Solutions**:
```python
# Tighten SCF convergence
"DM.Tolerance": "1.0e-5"

# Add more images
number_of_images = 7  # or 9

# Increase mesh cutoff
"Mesh.Cutoff": "350 Ry"

# Ensure consistent k-points
# Use same grid for all images
```

### Issue 6: "Multiple barriers in path"

**Symptoms**:
- Energy profile shows multiple peaks
- Unclear which is the true TS

**Interpretation**:
- May indicate multiple transition states
- Could be a complex reaction pathway
- Or sign of poor initial guess

**Solutions**:
```python
# Break into segments
# Run NEB for each barrier separately

# Use more images
number_of_images = 11  # Better resolution

# Consider alternative mechanisms
# Literature search for known pathways

# Use CI-NEB to refine highest barrier
"NEB.Climbing": "T"
```

---

## Advanced Techniques

### Climbing Image NEB

**After standard NEB converges**:

```python
# Enable climbing image
"NEB.Climbing": "T"

# Rerun NEB with climbing image
# Highest energy image will climb to true TS
```

**Benefits**:
- More accurate transition state
- Better barrier height
- True saddle point geometry

**Cost**: ~20-30% more iterations

### Variable Spring Constants

**Different springs for different regions**:

```lua
-- In custom neb.lua script
local k_spring = {}
k_spring[1] = 0.1  -- Weak near endpoints
k_spring[2] = 0.15
k_spring[3] = 0.20  -- Strong near TS
k_spring[4] = 0.15
k_spring[5] = 0.1
```

**Use when**: Very non-uniform energy landscape

### Free-End NEB

**Allow endpoints to relax**:
- Useful when exact endpoints unknown
- Endpoints move to nearest local minima
- More expensive

**Implementation**: Modify Lua script to relax endpoints

---

## Integration with Other Workflows

### Phonon Analysis of Transition State

```python
# After NEB, identify TS image
TS_structure = structures[TS_index]

# Run phonon calculation at TS
from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker
phonon_maker = SiestaPhononFlowMaker()
phonon_flow = phonon_maker.make(TS_structure)

# Should find ONE imaginary mode (reaction coordinate)
# All other modes should be real
```

**Validates TS**: True TS has exactly one imaginary frequency

### Thermodynamic Analysis

**Rate constant with zero-point correction**:
```python
from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker, SiestaQhaFlowMaker

# Calculate phonons at initial state and TS
phonon_initial = SiestaPhononFlowMaker().make(initial_structure)
phonon_TS = SiestaPhononFlowMaker().make(TS_structure)

# Extract vibrational free energy at temperature T
G_initial = phonon_initial.thermal_properties.free_energy[T]
G_TS = phonon_TS.thermal_properties.free_energy[T]

# Free energy barrier
ΔG‡ = G_TS - G_initial

# Rate constant (transition state theory)
k(T) = (k_B T / h) × exp(-ΔG‡ / k_B T)
```

**Includes**: Entropic contributions, zero-point energy corrections

---

## Next Steps

### After Completing This Tutorial

1. **Validate FLOS setup**:
   - Verify FLOS_PATH is correct
   - Test with minimal example
   - Check Lua script execution

2. **Run vacancy_diffusion example**:
   - Start with dry-run
   - Run locally with standard settings
   - Analyze energy profile

3. **Apply to your system**:
   - Define initial and final states carefully
   - Use converged parameters from convergence tutorials
   - Start with fewer images, refine as needed

4. **Validate results**:
   - Compare with literature if available
   - Check forces are converged
   - Verify energy profile is smooth
   - Consider CI-NEB refinement

### Related Tutorials

- **[01-convergence](../../01-convergence/)**: Essential parameter convergence before NEB
- **[04-mechanical](../../04-mechanical/)**: Elastic constants and mechanical properties
- **[06-vibrational-properties](../../06-vibrational-properties/)**: Phonons for TS characterization
- **[02-AseNebFlowMaker](../02-AseNebFlowMaker/)**: Alternative NEB implementation using ASE

---

## References

### Original NEB Papers

1. **Original NEB Method**:
   - Jónsson, H., Mills, G., & Jacobsen, K. W. (1998). "Nudged elastic band method for finding minimum energy paths of transitions". *Classical and Quantum Dynamics in Condensed Phase Simulations*.

2. **Climbing Image NEB**:
   - Henkelman, G., Uberuaga, B. P., & Jónsson, H. (2000). "A climbing image nudged elastic band method for finding saddle points and minimum energy paths". *J. Chem. Phys.*, 113, 9901.

3. **Improved Tangent**:
   - Henkelman, G., & Jónsson, H. (2000). "Improved tangent estimate in the nudged elastic band method for finding minimum energy paths and saddle points". *J. Chem. Phys.*, 113, 9978.

### SIESTA-Specific Resources

- [FLOS GitHub](https://github.com/siesta-project/flos)
- [FLOS NEB Examples](https://github.com/siesta-project/flos/tree/master/examples/neb)
- SIESTA Manual: NEB/FLOS section

### Online Resources

- [Wikipedia: Nudged Elastic Band](https://en.wikipedia.org/wiki/Nudged_elastic_band)
- [VASP NEB Tutorial](https://www.vasp.at/wiki/index.php/Nudged_Elastic_Band) (concepts transferable)

---

## Summary

**What we covered**:
- ✅ NEB theory and methodology
- ✅ Setting up NEB with FLOS/Lua
- ✅ Generating intermediate images
- ✅ Running NEB workflows
- ✅ Analyzing energy profiles
- ✅ Extracting activation barriers and transition states
- ✅ Troubleshooting common issues

**Key takeaways**:
1. Always relax endpoints first (local minima required)
2. Start with 5 images, add more if needed
3. Use converged parameters from convergence studies
4. Spring constant 0.1 eV/Å² is good starting point
5. Validate with dry-run before full calculation
6. CI-NEB improves transition state accuracy

**Ready for**: Production NEB calculations, transition state characterization, diffusion studies, reaction mechanism investigations

---

*Back to [05-barriers](../README.md) | [02-workflows](../../README.md) | [Main Tutorial Index](../../../README.md)*
