# Tutorial: FDF Block-Style Inputs

**Category**: 07-advanced-features
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), varies by example (local/submit)

---

## Overview

Learn to use SIESTA's FDF (Flexible Data Format) block-style inputs for advanced features like electric fields, custom k-point paths, geometry constraints, MD stress targets, PDOS projections, and spin initialization.

This consolidates 7 example scripts into a single configurable tutorial demonstrating all major FDF block types.

---

## What You'll Learn

- FDF block format and syntax
- List-based input specification in atomate2siesta
- External electric field application
- Custom k-point path definition (band structure)
- Geometry constraints (fix atoms, constrain directions)
- MD target stress specification
- PDOS (Projected Density of States) configuration
- Initial spin magnetization (DM.InitSpin)
- Combining multiple FDF blocks in one calculation

---

## Prerequisites

- **Required**: [01-relaxation](../../01-basics/01-relaxation/) completed
- **Recommended**: Basic understanding of SIESTA FDF format
- **Recommended**: Familiarity with k-point meshes and band structures

---

## Key Concepts

### FDF Block Format

**In SIESTA FDF files**:
```fdf
%block BlockName
  data line 1
  data line 2
  data line 3
%endblock BlockName
```

**In atomate2siesta (Python)**:
```python
fdf_arguments = {
    "BlockName": [
        "data line 1",
        "data line 2",
        "data line 3",
    ]
}
```

**Key Rule**: FDF blocks are specified as **lists of strings** in `fdf_arguments` dictionary.

### Common FDF Blocks

1. **ExternalElectricField**: Apply uniform electric field
2. **BandLinesScale**: Define custom k-point paths
3. **GeometryConstraints**: Fix atoms or constrain motion
4. **MD.TargetStress**: Target stress tensor for variable-cell MD
5. **ProjectedDensityOfStates**: PDOS energy range and resolution
6. **PDOS.kgrid_Monkhorst_Pack**: K-point mesh for PDOS
7. **DM.InitSpin**: Initial spin polarization per atom

---

## Configuration Options

### Example Types

#### 1. Electric Field (Graphene)
```python
EXAMPLE_TYPE = "electric_field"
```
- Apply perpendicular electric field (0.1 V/Å)
- Graphene bilayer example
- Field format: `Ex Ey Ez units`
- Enables electrostatic potential output

**Use Cases**:
- 2D materials under applied bias
- Polar materials and ferroelectrics
- Field-effect calculations

#### 2. Custom K-Point Path (Silicon)
```python
EXAMPLE_TYPE = "kpoint_path"
```
- Define high-symmetry path for band structure
- Silicon FCC example (Γ-X-W-K-Γ-L)
- Automatic path from high-symmetry-seekpath
- 40 points per segment

**Use Cases**:
- Non-standard Brillouin zones
- Specific k-point paths for analysis
- Custom band structure plots

#### 3. Geometry Constraints (Graphene Relaxation)
```python
EXAMPLE_TYPE = "constraints"
```
- Fix bottom 2 layers of graphene
- Allow top layers to relax
- Constraint format: `position from atom1 to atom2`

**Use Cases**:
- Surface relaxation (fix substrate)
- Defect calculations (fix boundaries)
- Selective atomic relaxation

#### 4. MD Target Stress (Silicon)
```python
EXAMPLE_TYPE = "md_stress"
```
- Variable-cell MD to achieve target stress
- Hydrostatic pressure (0.5 GPa)
- Stress tensor: 6 components (xx, yy, zz, xy, xz, yz)
- 10 ps MD with 2 fs timestep

**Use Cases**:
- Equilibrate under pressure
- NPT ensemble simulations
- Stress-strain calculations

#### 5. PDOS Projection (Silicon)
```python
EXAMPLE_TYPE = "pdos"
```
- Project DOS onto specific orbitals
- Energy range: -15 to +10 eV
- 0.1 eV resolution, 250 points
- Dedicated k-point mesh (6×6×6)

**Use Cases**:
- Orbital contribution analysis
- Band character identification
- Chemical bonding studies

#### 6. Initial Spin (Iron BCC)
```python
EXAMPLE_TYPE = "spin"
```
- Set initial spin polarization
- Ferromagnetic Fe: 2 μB per atom
- Accelerates SCF convergence
- Format: `atom_index spin_value`

**Use Cases**:
- Magnetic materials initialization
- Antiferromagnetic ordering
- Faster convergence for spin-polarized systems

#### 7. Multiple Blocks Combined (Comprehensive)
```python
EXAMPLE_TYPE = "multiple"
```
- Combines 4 FDF blocks simultaneously
- Electric field + constraints + spin + PDOS
- Demonstrates block compatibility
- Iron slab with complex setup

**Use Cases**:
- Production calculations
- Complex material setups
- Learning block interactions

---

## Quick Start

```bash
# 1. Preview electric field example
# Edit tutorial.py: EXAMPLE_TYPE = "electric_field", RUN_MODE = "dry_run"
python tutorial.py

# 2. Inspect generated FDF files
ls preview_output/job_*/inputs/
cat preview_output/job_*/inputs/input.fdf | grep -A5 "ExternalElectricField"

# 3. Try different block types
# Edit tutorial.py: EXAMPLE_TYPE = "kpoint_path"  # or others
python tutorial.py

# 4. Run actual calculation (quick: electric_field or constraints)
# Edit tutorial.py: RUN_MODE = "local"
python tutorial.py
```

---

## Expected Output

### Dry-Run Mode

```
✅ Dry-run complete!

📁 Preview files in: preview_output/job_<uuid>/inputs/

💡 Check input.fdf for FDF blocks:
  %block ExternalElectricField
    0.0 0.0 0.1 V/Ang
  %endblock ExternalElectricField
```

### Local/Submit Mode

**Electric Field**:
```
✅ Calculation complete!
📊 Output: Converged with electric field
💾 Electrostatic potential saved (if requested)
```

**K-Point Path**:
```
✅ Band structure calculation complete!
📊 Custom k-point path used
💡 Check .bands file for eigenvalues
```

**Constraints**:
```
✅ Relaxation complete with constraints!
📊 Bottom atoms fixed, top atoms relaxed
💡 Check forces (should be ~0 on fixed atoms)
```

**MD Stress**:
```
✅ MD simulation complete!
📊 Target stress: 0.5 GPa (hydrostatic)
💡 Check stress evolution in MD output
```

**PDOS**:
```
✅ PDOS calculation complete!
📊 Energy range: -15 to +10 eV
💡 Check .PDOS files for orbital projections
```

**Spin**:
```
✅ Spin-polarized calculation complete!
📊 Initial magnetization: 2 μB per Fe atom
💡 Check magnetic moments in output
```

---

## Understanding FDF Blocks

### Block 1: ExternalElectricField

**Purpose**: Apply uniform electric field to system

**Format**:
```python
fdf_arguments = {
    "ExternalElectricField": [
        "Ex Ey Ez units"  # Three components + unit
    ]
}
```

**Units**: `V/Ang`, `V/Bohr`, `Ry/Bohr/e`, `Har/Bohr/e`

**Example - Graphene**:
```python
# Perpendicular field (0.1 V/Å in z-direction)
"ExternalElectricField": ["0.0 0.0 0.1 V/Ang"]
```

**Important**:
- Field is uniform across entire cell
- For 2D materials, apply perpendicular to plane
- May require increased SCF iterations (`MaxSCFIterations: 200`)
- Consider saving electrostatic potential: `SaveElectrostaticPotential: True`

**Validation**:
- Check dipole moment change
- Verify band shifts (field × distance)
- Ensure SCF converges

---

### Block 2: BandLinesScale

**Purpose**: Define custom k-point path for band structure calculations

**Format**:
```python
fdf_arguments = {
    "BandLinesScale": ["ReciprocalLatticeVectors" or "pi/a"],
    "%block BandLines": [
        "1  kx1  ky1  kz1  Label1",
        "   kx2  ky2  kz2  Label2",
        "npoints kx3 ky3 kz3 Label3",
        # ...
    ]
}
```

**Example - Silicon FCC (Γ-X-W-K-Γ-L)**:
```python
"%block BandLines": [
    "1   0.000  0.000  0.000  \\Gamma",  # Start at Gamma
    "   40  0.500  0.000  0.500  X",     # 40 points to X
    "   40  0.500  0.250  0.750  W",     # 40 points to W
    "   40  0.375  0.375  0.750  K",     # 40 points to K
    "   40  0.000  0.000  0.000  \\Gamma", # Back to Gamma
    "   40  0.500  0.500  0.500  L",     # To L
]
```

**Important**:
- First line: number of bands + starting k-point
- Subsequent lines: number of points + ending k-point
- Coordinates in reciprocal lattice units (typically)
- Use `\\Gamma` for Γ symbol (LaTeX-like)

**Automation**:
```python
# Use high-symmetry-seekpath for automatic paths
from pymatgen.symmetry.bandstructure import HighSymmKpath

kpath = HighSymmKpath(structure)
# Or use atomate2's BandStructureMaker with auto_kpath=True
```

---

### Block 3: GeometryConstraints

**Purpose**: Fix atoms or constrain motion directions

**Format**:
```python
fdf_arguments = {
    "GeometryConstraints": [
        "constraint_type  specification"
    ]
}
```

**Constraint Types**:

**1. Fix Atoms (all directions)**:
```python
# Fix atoms 1-10
"GeometryConstraints": ["position from 1 to 10"]

# Fix specific atoms
"GeometryConstraints": ["position 1 5 7"]
```

**2. Fix Specific Directions**:
```python
# Fix x-direction for atoms 1-5
"GeometryConstraints": ["position 1 to 5 x"]

# Fix y,z for atom 3
"GeometryConstraints": ["position 3 y z"]
```

**3. Constrain Along Vector**:
```python
# Allow motion only along [1,1,0] direction
"GeometryConstraints": ["routine constr", "atom 1", "1.0 1.0 0.0"]
```

**Common Use Cases**:
- **Surface relaxation**: Fix bottom layers
- **Defect studies**: Fix boundary atoms
- **2D materials**: Fix in-plane motion for out-of-plane relaxation

**Example - Graphene on Substrate**:
```python
# 4 layers, fix bottom 2 (8 atoms)
n_fixed = 8
fdf_arguments = {
    "GeometryConstraints": [f"position from 1 to {n_fixed}"]
}
```

**Important**:
- Atom indices start at 1 (not 0!)
- Constrained atoms still included in SCF
- Forces on fixed atoms should be ~0 after relaxation

---

### Block 4: MD.TargetStress

**Purpose**: Target stress tensor for variable-cell MD (NPT ensemble)

**Format**:
```python
fdf_arguments = {
    "MD.TargetStress": [
        "Sxx Syy Szz Sxy Sxz Syz units"  # 6 components
    ]
}
```

**Units**: `GPa`, `Ry/Bohr**3`, `eV/Ang**3`

**Examples**:

**1. Hydrostatic Pressure (0.5 GPa)**:
```python
"MD.TargetStress": ["0.5 0.5 0.5 0.0 0.0 0.0 GPa"]
# Equal normal stresses, zero shear
```

**2. Uniaxial Stress (1 GPa in z)**:
```python
"MD.TargetStress": ["0.0 0.0 1.0 0.0 0.0 0.0 GPa"]
```

**3. Shear Stress**:
```python
"MD.TargetStress": ["0.0 0.0 0.0 0.5 0.0 0.0 GPa"]  # xy shear
```

**Important**:
- Requires `MD.VariableCell: True`
- Use with Parrinello-Rahman or similar algorithms
- Stress components: (xx, yy, zz, xy, xz, yz)
- Negative values = compression (convention depends on SIESTA version)

**Typical MD Settings**:
```python
fdf_arguments = {
    "MD.TypeOfRun": "Verlet",
    "MD.VariableCell": True,
    "MD.NumCGsteps": 100,
    "MD.MaxForceTol": "0.02 eV/Ang",
    "MD.MaxStressTol": "0.1 GPa",
    "MD.TargetStress": ["0.5 0.5 0.5 0.0 0.0 0.0 GPa"],
}
```

---

### Block 5: ProjectedDensityOfStates (PDOS)

**Purpose**: Calculate orbital-projected density of states

**Format**:
```python
fdf_arguments = {
    "ProjectedDensityOfStates": [
        "Emin Emax dE nE units"
    ]
}
```

**Parameters**:
- `Emin`: Lower energy bound
- `Emax`: Upper energy bound
- `dE`: Energy resolution (broadening)
- `nE`: Number of points
- `units`: `eV`, `Ry`, `Har`

**Example - Silicon PDOS**:
```python
fdf_arguments = {
    "ProjectedDensityOfStates": ["-15.0  10.0  0.1  250  eV"],
    # -15 to +10 eV, 0.1 eV resolution, 250 points
}
```

**Dedicated K-Point Mesh**:
```python
fdf_arguments = {
    "ProjectedDensityOfStates": ["-15.0  10.0  0.1  250  eV"],
    "PDOS.kgrid_Monkhorst_Pack": [
        "6  0  0  0.0",  # 6 divisions in a-direction
        "0  6  0  0.0",  # 6 divisions in b-direction
        "0  0  6  0.0",  # 6 divisions in c-direction
    ]
}
```

**Output Files**:
- `SystemLabel.PDOS` - Total DOS
- `SystemLabel.PDOS.n.orbital` - Orbital projections (e.g., 1s, 2p)

**Important**:
- Requires dense k-point mesh (often denser than SCF)
- Use `PDOS.kgrid_Monkhorst_Pack` for separate PDOS mesh
- Energy range should cover valence + conduction bands
- Resolution trade-off: smaller dE = smoother, more points

**Analysis**:
```bash
# Plot total DOS
grep -v "#" SystemLabel.PDOS | awk '{print $1, $2}' > dos.dat

# Plot s-orbital contribution
grep -v "#" SystemLabel.PDOS.1.s | awk '{print $1, $2}' > s_dos.dat
```

---

### Block 6: DM.InitSpin

**Purpose**: Set initial spin polarization per atom (magnetic systems)

**Format**:
```python
fdf_arguments = {
    "DM.InitSpin": [
        "atom_index  spin_value",  # Repeat for each atom
        # ...
    ]
}
```

**Units**: Bohr magnetons (μB)

**Example - Ferromagnetic Iron (2 atoms)**:
```python
fdf_arguments = {
    "Spin": "polarized",  # Enable spin polarization
    "DM.InitSpin": [
        "1  2.0",  # Atom 1: +2 μB (spin up)
        "2  2.0",  # Atom 2: +2 μB (spin up)
    ]
}
```

**Example - Antiferromagnetic**:
```python
fdf_arguments = {
    "Spin": "polarized",
    "DM.InitSpin": [
        "1   2.0",   # Up
        "2  -2.0",   # Down
        "3   2.0",   # Up
        "4  -2.0",   # Down
    ]
}
```

**Programmatic Generation**:
```python
# All atoms spin up
n_atoms = len(structure)
fdf_arguments = {
    "Spin": "polarized",
    "DM.InitSpin": [f"{i+1}  2.0" for i in range(n_atoms)]
}
```

**Important**:
- Requires `Spin: polarized` or `Spin: spin-orbit`
- Initial guess only (converges to self-consistent solution)
- Accelerates convergence for magnetic systems
- Large values may cause SCF instability (use ~actual magnetic moment)

**Validation**:
```bash
# Check converged magnetic moments in output
grep "Total spin polarization" output.out
grep "Magnetic moment per atom" output.out
```

---

### Block 7: Multiple Blocks Combined

**Purpose**: Use several FDF blocks in one calculation

**Example - Iron Slab (Complex Setup)**:
```python
n_atoms = len(structure)
n_fixed = 4  # Bottom layer

fdf_arguments = {
    # Electric field
    "ExternalElectricField": ["0.0 0.0 0.1 V/Ang"],

    # Fix bottom layer
    "GeometryConstraints": [f"position from 1 to {n_fixed}"],

    # Initialize spin (only free atoms)
    "Spin": "polarized",
    "DM.InitSpin": [f"{i+1}  2.0" for i in range(n_fixed, n_atoms)],

    # PDOS projection
    "ProjectedDensityOfStates": ["-15.0  10.0  0.1  250  eV"],
    "PDOS.kgrid_Monkhorst_Pack": [
        "6  0  0  0.0",
        "0  6  0  0.0",
        "0  0  6  0.0",
    ],
}
```

**Compatibility Considerations**:
- ✅ **Compatible**: Constraints + Electric field
- ✅ **Compatible**: Spin + PDOS
- ✅ **Compatible**: Most blocks can coexist
- ⚠️ **Careful**: BandLines + PDOS (both define k-points, separate calculations)
- ⚠️ **Careful**: MD.TargetStress + GeometryConstraints (constraints limit cell motion)

**Best Practices**:
1. Test blocks individually first
2. Combine incrementally
3. Check for parameter conflicts
4. Verify all blocks appear in `input.fdf`

---

## Debugging FDF Blocks

### Verification Steps

**1. Check Generated FDF File**:
```bash
# Dry-run first
python tutorial.py  # RUN_MODE = "dry_run"

# Inspect input.fdf
cat preview_output/job_*/inputs/input.fdf
```

**2. Look for Block Syntax**:
```bash
# Should see:
%block ExternalElectricField
  0.0 0.0 0.1 V/Ang
%endblock ExternalElectricField
```

**3. Common Errors**:

**Error: Block not appearing in FDF**
```python
# WRONG: String instead of list
fdf_arguments = {
    "ExternalElectricField": "0.0 0.0 0.1 V/Ang"  # ❌
}

# CORRECT: List of strings
fdf_arguments = {
    "ExternalElectricField": ["0.0 0.0 0.1 V/Ang"]  # ✅
}
```

**Error: Syntax in block content**
```python
# WRONG: Missing units
"ExternalElectricField": ["0.0 0.0 0.1"]  # ❌

# CORRECT: Include units
"ExternalElectricField": ["0.0 0.0 0.1 V/Ang"]  # ✅
```

**Error: Atom indexing**
```python
# WRONG: Python 0-indexing
"GeometryConstraints": ["position from 0 to 7"]  # ❌

# CORRECT: SIESTA 1-indexing
"GeometryConstraints": ["position from 1 to 8"]  # ✅
```

### Validation Techniques

**1. Electric Field**: Check dipole moment change
```bash
grep "Electric dipole" output.out
# Should show field-induced polarization
```

**2. Constraints**: Check forces on fixed atoms
```bash
grep "Forces" output.out
# Fixed atoms should have ~0 force
```

**3. Spin**: Verify initial vs. final magnetization
```bash
grep -A2 "InitSpin" output.out
grep "Total magnetic moment" output.out
```

**4. PDOS**: Check output files exist
```bash
ls *.PDOS*
# Should see: SystemLabel.PDOS, SystemLabel.PDOS.1.s, etc.
```

---

## Common Issues

### Issue 1: FDF Block Not Recognized

**Symptoms**: Block doesn't appear in generated `input.fdf`

**Solutions**:
1. Ensure block name is a **list of strings**:
   ```python
   # CORRECT
   fdf_arguments = {"ExternalElectricField": ["0.0 0.0 0.1 V/Ang"]}
   ```

2. Check spelling (case-sensitive):
   ```python
   # WRONG: externalelectricfield
   # CORRECT: ExternalElectricField
   ```

3. Verify in `fdf_arguments` dictionary (not other params)

4. Check atomate2siesta version supports block

### Issue 2: SCF Not Converging with Electric Field

**Symptoms**: SCF cycles exceed max iterations

**Solutions**:
1. Increase max iterations:
   ```python
   fdf_arguments = {"MaxSCFIterations": 200}  # Default: 50
   ```

2. Reduce electric field strength (start small: 0.01 V/Å)

3. Tighten DM tolerance:
   ```python
   fdf_arguments = {"DM.Tolerance": 1.0e-5}  # Default: 1e-4
   ```

4. Use mixing schemes:
   ```python
   fdf_arguments = {
       "SCF.Mixer.Method": "Pulay",
       "SCF.Mixer.Weight": 0.1,
   }
   ```

### Issue 3: Constraints Not Working

**Symptoms**: "Fixed" atoms still moving

**Solutions**:
1. Check atom indexing (1-based, not 0-based!)
   ```python
   # For Python list indices 0-7, use SIESTA indices 1-8
   "GeometryConstraints": ["position from 1 to 8"]
   ```

2. Verify constraint appears in FDF:
   ```bash
   grep -A2 "GeometryConstraints" input.fdf
   ```

3. Check relaxation type (CG, Broyden support constraints)

4. Inspect forces (should be ~0 on constrained atoms)

### Issue 4: PDOS Files Empty or Missing

**Symptoms**: No `.PDOS` files generated

**Solutions**:
1. Ensure PDOS block syntax correct:
   ```python
   "ProjectedDensityOfStates": ["-15.0  10.0  0.1  250  eV"]
   # Five space-separated values + unit
   ```

2. Check k-point mesh is adequate:
   ```python
   "PDOS.kgrid_Monkhorst_Pack": [
       "6  0  0  0.0",
       "0  6  0  0.0",
       "0  0  6  0.0",
   ]
   ```

3. Verify calculation completed successfully

4. Check energy range includes relevant states

### Issue 5: Spin Initialization Ignored

**Symptoms**: Calculation converges to non-magnetic solution

**Solutions**:
1. Enable spin polarization:
   ```python
   fdf_arguments = {"Spin": "polarized"}  # REQUIRED!
   ```

2. Check initial spin values are reasonable (~actual magnetic moment)

3. May need to constrain total spin:
   ```python
   "FixSpin": True,
   "TotalSpin": 4.0,  # Total magnetic moment
   ```

4. Verify system is actually magnetic (check literature)

---

## Best Practices

### FDF Block Usage

**1. Start Simple**:
- Test one block at a time
- Verify in dry-run mode first
- Check generated `input.fdf` carefully

**2. Use Dry-Run for Debugging**:
```python
RUN_MODE = "dry_run"
# Always inspect preview_output/job_*/inputs/input.fdf
```

**3. List Format is Critical**:
```python
# ALWAYS use lists for FDF blocks
fdf_arguments = {
    "BlockName": [
        "line 1",
        "line 2",
    ]
}
```

**4. Combine Logically**:
- Group related blocks (e.g., PDOS + PDOS k-grid)
- Avoid conflicting settings (e.g., constraints + free cell MD)
- Document complex combinations

### Parameter Selection

**Electric Field**:
- 2D materials: 0.01-0.5 V/Å typical
- 3D materials: Smaller fields (0.001-0.1 V/Å)
- Increase SCF iterations if needed

**Constraints**:
- Surface: Fix bottom 1-2 layers
- Defects: Fix atoms > 10 Å from defect
- Always verify force convergence

**MD Stress**:
- Start with hydrostatic (equal diagonal components)
- Use realistic pressures (< 10 GPa for most materials)
- Monitor stress evolution during MD

**PDOS**:
- Energy range: Fermi ± 15-20 eV typical
- Resolution: 0.1 eV for metals, 0.05 eV for insulators
- K-mesh: Denser than SCF (often 2× or more)

**Spin**:
- Initialize close to expected moment
- Ferromagnetic: All same sign
- Antiferromagnetic: Alternating signs

### Documentation

**Comment Your Blocks**:
```python
fdf_arguments = {
    # Apply perpendicular electric field (0.1 V/Å)
    "ExternalElectricField": ["0.0 0.0 0.1 V/Ang"],

    # Fix bottom 8 atoms (substrate layer)
    "GeometryConstraints": ["position from 1 to 8"],

    # PDOS: -15 to +10 eV, 0.1 eV resolution
    "ProjectedDensityOfStates": ["-15.0  10.0  0.1  250  eV"],
}
```

---

## Tips for Success

✅ **Always dry-run first**: Check generated FDF before expensive calculations

✅ **List format for blocks**: FDF blocks must be lists of strings

✅ **1-based atom indexing**: SIESTA atoms start at 1, not 0

✅ **Include units**: Electric field, stress, PDOS energy all need units

✅ **Test individually**: Verify each block works alone before combining

✅ **Check output files**: Ensure expected files (.PDOS, .bands, etc.) are created

✅ **Start with defaults**: Use SIESTA manual recommended values initially

✅ **Validate physically**: Electric field effects, constraint forces, spin moments should be reasonable

---

## Advanced Usage

### Dynamic Block Generation

**Example: Programmatic Constraint Creation**
```python
from pymatgen.core import Structure

structure = Structure(...)
n_atoms = len(structure)

# Fix atoms with z < 5 Å
fixed_indices = [
    i+1  # SIESTA 1-indexing
    for i, site in enumerate(structure)
    if site.coords[2] < 5.0
]

# Create constraint string
if len(fixed_indices) > 0:
    constraint = f"position {' '.join(map(str, fixed_indices))}"
    fdf_arguments = {"GeometryConstraints": [constraint]}
```

### Combining with Powerups

```python
from atomate2.siesta.powerups import update_user_siesta_settings
from atomate2.siesta.jobs.core import RelaxMaker

# Create job
maker = RelaxMaker.fixed_cell_relaxation()
job = maker.make(structure)

# Add FDF blocks via powerup
fdf_blocks = {
    "ExternalElectricField": ["0.0 0.0 0.1 V/Ang"],
    "GeometryConstraints": ["position from 1 to 4"],
}
job = update_user_siesta_settings(job, fdf_blocks)
```

### Multi-Step Workflows with Blocks

```python
from jobflow import Flow
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker

# Step 1: Relaxation with constraints
relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "GeometryConstraints": ["position from 1 to 4"],
    }
)
relax_job = relax_maker.make(structure)

# Step 2: PDOS with electric field
static_maker = StaticMaker(
    user_params={
        "ExternalElectricField": ["0.0 0.0 0.1 V/Ang"],
        "ProjectedDensityOfStates": ["-15.0  10.0  0.1  250  eV"],
    }
)
pdos_job = static_maker.make(relax_job.output.structure)

# Combine in flow
flow = Flow([relax_job, pdos_job])
```

---

## Reference: Complete FDF Block List

**Geometry & Constraints**:
- `GeometryConstraints` - Fix atoms or directions
- `Geometry.Constraints` - Alternative syntax

**Electronic Structure**:
- `ExternalElectricField` - Applied electric field
- `DM.InitSpin` - Initial spin polarization
- `Spin.Fix` - Constrain total spin

**K-Points & Bands**:
- `BandLines` - Custom k-point path
- `BandLinesScale` - Coordinate system
- `PDOS.kgrid_Monkhorst_Pack` - PDOS k-mesh
- `kgrid_Monkhorst_Pack` - SCF k-mesh (alternative to 3-integer format)

**Density of States**:
- `ProjectedDensityOfStates` - PDOS parameters
- `LocalDensityOfStates` - LDOS (real-space)

**Molecular Dynamics**:
- `MD.TargetStress` - Target stress tensor
- `MD.TargetPressure` - Target pressure (scalar)

**Advanced**:
- `WaveFuncKPoints` - Specific k-points for wavefunctions
- `COOP.Write` - Crystal orbital overlap population
- `TS.Contour.Eta` - Non-equilibrium contours (TranSIESTA)

**See SIESTA manual** (`siesta.pdf`) for complete list and syntax.

---

## Next Steps

After completing this tutorial:

1. **Test all 7 examples**: Understand each block type individually
2. **Combine blocks**: Try `multiple` example, create custom combinations
3. **Read SIESTA manual**: Deep dive into block syntax and options
4. **Apply to research**: Use FDF blocks in production calculations
5. **Advanced blocks**: Explore TranSIESTA, COOP, LDOS blocks
6. **Troubleshoot**: Practice debugging FDF syntax and validation

---

## Further Reading

- **SIESTA Manual**: Complete FDF block reference (`siesta.pdf`)
- **atomate2siesta Docs**: Parameter organization and dataclass hierarchy
- **Tutorial 01-relaxation**: Basic parameter passing
- **Tutorial 03-band-structure**: K-point paths and band calculations
- **Tutorial 18-tier-based-calculations**: Preset parameter sets

---

*Back to [07-advanced-features](../README.md) | [Main Tutorial Index](../../README.md)*
