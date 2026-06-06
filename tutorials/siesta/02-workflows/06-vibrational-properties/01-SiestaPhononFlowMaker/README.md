# Tutorial: Phonon Calculations

**Category**: 06-vibrational-properties
**Difficulty**: Advanced
**Time**: ~5 min (dry-run), ~2-6 hours (full calculation)

---

## Overview

Complete phonon calculation workflow using phonopy integration with SIESTA. Computes phonon band structure, density of states, and thermal properties from first principles.

This tutorial demonstrates the **SiestaPhononFlowMaker**, which automates supercell generation, displacement calculations, force constant extraction, and phonon analysis with automatic plotting.

---

## What You'll Learn

- Phonon calculation workflow using finite displacements
- Supercell generation (automatic and manual)
- Separate parameters for relaxation vs force calculations
- Phonon band structure and density of states
- Thermal properties (heat capacity, entropy, free energy)
- Automatic plotting with phonopy
- Critical importance of well-relaxed structures

---

## Prerequisites

- **Required**: [01-RelaxMaker](../../../01-basics/01-RelaxMaker/) completed
- **CRITICAL**: **Fully relaxed structure** with forces < 0.01 eV/Å
- **Required**: [01-convergence](../../01-convergence/) - Converged k-points and basis
- **Recommended**: Basic phonon theory and phonopy familiarity

---

## Key Concepts

### Phonon Calculation Method

**Finite Displacement Approach** (phonopy):
1. Create supercell (e.g., 2×2×2) from unit cell
2. Apply small atomic displacements (0.01 Å typical)
3. Calculate forces on all atoms for each displacement
4. Extract force constants from force-displacement relationship
5. Diagonalize dynamical matrix → phonon frequencies and eigenvectors
6. Plot band structure, DOS, thermal properties

**Force Constant Matrix**:
For atoms i and j in Cartesian directions α and β:

$
\Phi_{i\alpha,j\beta} = \frac{\partial^2 E}{\partial u_{i\alpha} \partial u_{j\beta}} \approx \frac{F_{j\beta}(u_{i\alpha})}{u_{i\alpha}}
$

Where u is displacement, F is force.

### Supercell Requirements

**Size Considerations**:
- **Minimum**: 2×2×2 (quick testing)
- **Standard**: 3×3×3 (publication quality)
- **Very large systems**: 2×2×2 may suffice

**Two Specification Methods**:
1. **Automatic** (`min_length`): Phonopy chooses supercell ≥ specified length
2. **Explicit** (`supercell_matrix`): Full control over supercell shape

**Rule**: Supercell should be large enough that force constants decay to zero at boundaries (typically > 12 Å)

### Number of Displacement Calculations

Depends on symmetry:
- **High symmetry** (cubic): 6-10 displacements
- **Medium symmetry** (hexagonal): 10-20 displacements
- **Low symmetry** (triclinic): 50-100+ displacements

Use `dry_run=True` to preview number before running!

### Separate Relaxation and Force Parameters

**Strategy** (demonstrated in `01_standard.py`):
- **Relaxation**: Moderate parameters (faster, just need geometry)
  - Looser k-points [3,3,3]
  - Standard cutoff (300 Ry)
  - Moderate basis (DZP)

- **Force calculations**: Tight parameters (accurate forces critical!)
  - Denser k-points [6,6,6] or higher
  - Higher cutoff (400-500 Ry)
  - Tight SCF (DM.Tolerance=1e-6)

**Why**: Phonon frequencies very sensitive to force accuracy, but not to initial structure optimization details.

---

## Workflow Structure

```
SiestaPhononFlowMaker
├── Variable-cell relaxation
│   └── Get equilibrium structure
├── Generate supercell
│   └── Based on min_length or supercell_matrix
├── Generate displacements
│   └── Phonopy creates symmetry-reduced list
├── Displacement calculations (many jobs!)
│   ├── Displacement 1: Static calculation → forces
│   ├── Displacement 2: Static calculation → forces
│   ├── ...
│   └── Displacement N: Static calculation → forces
└── Phonon analysis
    ├── Extract force constants
    ├── Compute phonon band structure
    ├── Compute phonon DOS
    ├── Compute thermal properties
    └── Automatic plotting (3 PNG files)
```

---

## Quick Start

### Basic Example (Automatic Supercell)

```python
from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker
from pymatgen.core import Structure
from jobflow import run_locally

# Load relaxed structure (CRITICAL: forces < 0.01 eV/Å!)
structure = Structure.from_file("relaxed_structure.cif")

# Create phonon workflow
flow = SiestaPhononFlowMaker(
    min_length=12.0,  # Automatic supercell ≥ 12 Å
    displacement=0.01,  # 0.01 Å displacement
    use_symmetry=True,  # Reduce displacements using symmetry
    mesh=(50, 50, 50),  # Dense q-point mesh for DOS
    create_thermal_properties=True,  # Calculate Cv, entropy, F
    t_min=0,
    t_max=1000,
    t_step=10,
)

# Run
workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True)
```

### Explicit Supercell with Custom Parameters

```python
from atomate2.siesta.jobs.core import StaticMaker, LuaMaker

# Relaxation parameters (moderate)
relax_params = {
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [3, 3, 3],
    "Mesh.Cutoff": "300 Ry",
    "MD.MaxForceTol": "0.01 eV/Ang",
}

# Force calculation parameters (tight!)
force_params = {
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [6, 6, 6],  # Denser k-points
    "Mesh.Cutoff": "450 Ry",  # Higher cutoff
    "DM.Tolerance": 1e-6,  # Tighter SCF
}

# Create makers with custom parameters
relax_maker = LuaMaker.variable_cell_relaxation(
    use_custodian=True,
    user_params=relax_params,
)

static_maker = StaticMaker.scf(
    use_custodian=True,
    user_params=force_params,
)

# Create phonon workflow
flow = SiestaPhononFlowMaker(
    relax_maker=relax_maker,  # Custom relaxation
    static_maker=static_maker,  # Custom force calculations
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],  # Explicit 2×2×2
    displacement=0.01,
    use_symmetry=True,
    mesh=(50, 50, 50),
    create_thermal_properties=True,
    t_min=0,
    t_max=1000,
    t_step=10,
)

workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True)
```

---

## Configuration Options

### Supercell Specification

```python
# Method 1: Automatic (phonopy chooses)
min_length=12.0  # Å

# Method 2: Explicit control
supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]]  # 2×2×2

# Method 3: Non-cubic (layered materials)
supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 2]]  # 3×3×2 (sparser in z)
```

### Displacement Distance

```python
# Standard (recommended)
displacement=0.01  # Å

# Larger (faster convergence, less accurate)
displacement=0.02  # Å

# Smaller (more accurate, noisier with numerical errors)
displacement=0.005  # Å
```

### Q-Point Mesh for DOS

```python
# Coarse (fast)
mesh=(20, 20, 20)

# Standard (recommended)
mesh=(50, 50, 50)

# Dense (high accuracy)
mesh=(100, 100, 100)
```

### Thermal Properties Temperature Range

```python
# Standard range
t_min=0      # K
t_max=1000   # K
t_step=10    # K

# High temperature
t_min=0
t_max=2000   # K (check QHA validity!)
t_step=20
```

---

## Output

### File Structure

```
job_phonon_analysis_*/
├── phonon_bands.png              # Phonon band structure
├── phonon_dos.png                # Phonon density of states
├── thermal_properties.png        # Cv, entropy, F vs T
├── phonon_summary.txt            # Text summary
├── phonopy.yaml                  # Phonopy output (full data)
├── force_constants.yaml          # Force constant matrix
└── band.yaml                     # Band structure data
```

### Phonon Summary

```python
with open("job_phonon_analysis_*/phonon_summary.txt") as f:
    print(f.read())
```

**Output**:
```
Phonon Calculation Summary
==========================

Structure: Si2
Supercell: 2×2×2 (16 atoms)
Displacements: 6 (symmetry-reduced)

Phonon Properties:
  Minimum frequency:  0.00 cm⁻¹ (acoustic mode at Γ)
  Maximum frequency: 524.13 cm⁻¹
  No imaginary frequencies ✓

Thermal Properties (300 K):
  Helmholtz free energy: -0.152 eV/atom
  Entropy:                0.00051 eV/K/atom
  Heat capacity (Cv):     0.00034 eV/K/atom

Zero-point energy: 0.032 eV/atom
```

### Analyzing Results

```python
import yaml
import numpy as np

# Read phonopy output
with open("job_phonon_analysis_*/phonopy.yaml") as f:
    phonopy_data = yaml.safe_load(f)

# Extract phonon frequencies
frequencies = []
for qpoint in phonopy_data['phonon']:
    for band in qpoint['band']:
        freq = band['frequency']  # cm⁻¹
        frequencies.append(freq)

frequencies = np.array(frequencies)

# Check for imaginary modes (negative frequencies)
imaginary = frequencies[frequencies < 0]
if len(imaginary) > 0:
    print(f"WARNING: {len(imaginary)} imaginary frequencies!")
    print(f"Minimum frequency: {frequencies.min():.2f} cm⁻¹")
else:
    print("✓ No imaginary frequencies - structure is stable")
```

---

## Best Practices

✅ **CRITICAL - Well-relaxed structure**: Forces < 0.01 eV/Å before phonon calculation
✅ **Converged parameters**: Use converged k-points, cutoff, basis from convergence studies
✅ **Separate relaxation/force params**: Tight parameters for forces, moderate for relaxation
✅ **Use symmetry**: `use_symmetry=True` reduces calculations dramatically
✅ **Preview first**: `dry_run=True` to see number of displacements
✅ **Enable custodian**: Automatic error recovery for long calculations
✅ **Dense q-mesh**: mesh=(50,50,50) or higher for smooth DOS

❌ **Don't use unrelaxed structure**: Will give imaginary frequencies
❌ **Don't use too small supercell**: < 10 Å risks artificial interactions
❌ **Don't skip convergence**: Phonons very sensitive to parameters
❌ **Don't ignore imaginary modes**: Indicates structural instability or insufficient relaxation

---

## Importance of Relaxed Structure

**Why forces < 0.01 eV/Å is critical**:

```python
# ❌ WRONG - Using unrelaxed structure
structure = Structure.from_file("unrelaxed.cif")  # Forces = 0.15 eV/Å
phonon_flow = SiestaPhononFlowMaker().make(structure)
# Result: Many imaginary frequencies!

# ✅ CORRECT - Relax first
from atomate2.siesta.jobs.core import LuaMaker

relax_job = LuaMaker.variable_cell_relaxation(
    user_params={"MD.MaxForceTol": "0.005 eV/Ang"}  # Very tight!
).make(structure)

# Then use relaxed structure for phonons
phonon_flow = SiestaPhononFlowMaker().make(relax_job.output.structure)
```

**Physical reason**: Small residual forces → large spurious force constants → wrong frequencies

---

## Supercell Size Convergence

Test phonon convergence with supercell size:

```python
import matplotlib.pyplot as plt

supercells = [
    [[2,0,0],[0,2,0],[0,0,2]],  # 2×2×2
    [[3,0,0],[0,3,0],[0,0,3]],  # 3×3×3
    [[4,0,0],[0,4,0],[0,0,4]],  # 4×4×4
]

max_frequencies = []

for sc in supercells:
    flow = SiestaPhononFlowMaker(supercell_matrix=sc)
    # Run and extract maximum frequency
    # max_freq = ...
    max_frequencies.append(max_freq)

plt.plot([2,3,4], max_frequencies, 'o-')
plt.xlabel("Supercell size (n in n×n×n)")
plt.ylabel("Maximum frequency (cm⁻¹)")
plt.title("Phonon convergence vs supercell size")

# Converged when change < 1-2 cm⁻¹
```

**Typical result**: 2×2×2 often sufficient for cubic systems, 3×3×3 for high accuracy

---

## Troubleshooting

**Problem**: Imaginary frequencies (negative values)

**Solution**:
1. **Most common**: Structure not fully relaxed
   - Re-relax with MD.MaxForceTol="0.005 eV/Ang"
   - Check final forces < 0.01 eV/Å
2. **Structure actually unstable**: Phonons correctly predict instability
   - Check for phase transitions
   - Try different structure (temperature, pressure)
3. **Supercell too small**: Artificial interactions
   - Increase supercell size (2×2×2 → 3×3×3)

---

**Problem**: Too many displacement calculations (> 100)

**Solution**:
1. Enable symmetry: `use_symmetry=True` (default)
2. Use smaller supercell for testing: 2×2×2 instead of 3×3×3
3. Check structure has correct symmetry:
   ```python
   from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
   sga = SpacegroupAnalyzer(structure)
   print(f"Space group: {sga.get_space_group_symbol()}")
   ```
4. Submit to HPC cluster with job arrays

---

**Problem**: Phonon frequencies don't match experiments

**Solution**:
1. **Most common**: Unconverged k-points or basis
   - Run convergence tests first!
   - Phonons very sensitive to computational parameters
2. Check supercell size: Try 3×3×3
3. Tighter force calculation parameters:
   ```python
   force_params = {
       "a2s_kpts": [8, 8, 8],  # Very dense
       "Mesh.Cutoff": "500 Ry",  # Very high
       "DM.Tolerance": 1e-7,  # Very tight
   }
   ```
4. GGA limitations: PBE typically overestimates frequencies by ~2-5%

---

**Problem**: Calculation takes too long

**Solution**:
1. Use smaller supercell for testing (2×2×2)
2. Enable custodian: Automatic error recovery prevents restarts
3. Submit to HPC cluster
4. Check number of displacements: `dry_run=True` first

---

## Thermal Properties Interpretation

### Heat Capacity at Constant Volume (Cv)

```python
# Read thermal properties
with open("job_phonon_analysis_*/thermal_properties.yaml") as f:
    thermal = yaml.safe_load(f)

temperatures = [entry['temperature'] for entry in thermal['thermal_properties']]
cv_values = [entry['heat_capacity'] for entry in thermal['thermal_properties']]

# Classical limit (Dulong-Petit): Cv → 3kB per atom
classical_cv = 3 * 8.617e-5  # eV/K/atom

print(f"Cv at 300 K: {cv_values[30]:.5f} eV/K/atom")
print(f"Classical limit: {classical_cv:.5f} eV/K/atom")
print(f"Ratio: {cv_values[30]/classical_cv:.2f}")
```

**Expected**: Cv → 3kB at high T (Dulong-Petit law)

### Zero-Point Energy

Quantum contribution at T=0:

$
E_{ZPE} = \sum_{\mathbf{q},s} \frac{1}{2} \hbar \omega_{\mathbf{q},s}
$

Typically 0.01-0.1 eV/atom depending on element mass.

---

## Next Steps

After completing phonon calculations:

1. **Grüneisen parameters**: [02-SiestaGruneisenFlowMaker](../02-SiestaGruneisenFlowMaker/) - Volume-dependent phonons
2. **QHA thermodynamics**: [03-SiestaQhaFlowMaker](../03-SiestaQhaFlowMaker/) - Temperature-dependent free energy
3. **Thermal conductivity**: Use phonon results as input for transport calculations
4. **Spectroscopy**: Compare with IR/Raman experiments

---

## Related Tutorials

- [02-SiestaGruneisenFlowMaker](../02-SiestaGruneisenFlowMaker/) - Grüneisen parameters
- [03-SiestaQhaFlowMaker](../03-SiestaQhaFlowMaker/) - Quasi-harmonic approximation
- [Vibrational Properties Overview](../README.md) - All phonon tutorials
- [01-convergence](../../01-convergence/) - Parameter convergence (do this first!)

---

**📚 [Back to Vibrational Properties](../README.md)** | **📖 [All Tutorials](../../../README.md)**
