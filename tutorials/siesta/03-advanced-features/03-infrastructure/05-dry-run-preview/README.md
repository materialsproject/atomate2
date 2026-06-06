# Dry-Run Mode Tutorial

## Overview

**Dry-run mode** is a powerful feature that allows you to preview workflows and generate complete SIESTA input files **without running expensive calculations**. This is essential for:

- **Validation**: Verify workflow setup before committing computational resources
- **Parameter Inspection**: Check all SIESTA parameters in generated input files
- **Structure Preview**: Visualize transformed structures (relaxation, EOS, surfaces)
- **Time Savings**: Convergence studies can be validated in seconds instead of hours
- **Zero Cost**: No SIESTA calculations are performed

### When to Use Dry-Run Mode

✅ **Before running expensive workflows**:
- Convergence studies (k-points, mesh cutoff, basis parameters)
- EOS calculations with multiple volumes
- Surface energy calculations
- NEB transition state searches
- Phonon calculations with large supercells

✅ **When learning atomate2siesta**:
- Understand how workflows transform structures
- Learn SIESTA parameter organization
- Explore different calculation types risk-free

✅ **For workflow development**:
- Test new workflow combinations
- Verify parameter propagation
- Debug complex multi-step workflows

---

## Key Concepts

### 1. What Dry-Run Generates

Each dry-run job creates a **complete SIESTA input package**:

```
output_directory/
├── structure.cif              # Structure file (CIF, XSF, JSON, POSCAR)
├── siesta.fdf                 # Main SIESTA input with ALL parameters
├── structure.fdf              # Structure definition (geometry)
├── Si.psml                    # Pseudopotential file(s)
├── siesta_parameters.json     # Parameters in JSON format
└── workflow_summary.txt       # Human-readable summary
```

**Key file: `siesta.fdf`** - Contains complete calculation setup:
- All FDF parameters (basis, k-points, mesh cutoff, SCF, etc.)
- PAO.BasisSize, OccupationFunction, electronic temperature
- Convergence criteria, mixing parameters
- Output options

### 2. Automatic Propagation in Flows

When you set `dry_run=True` at the **flow level**, it automatically propagates to **all child makers**:

```python
# ✨ Set dry_run ONCE at flow level
flow = KpointsConvergenceMaker(
    kpoints_list=[[2,2,2], [4,4,4], [6,6,6]],
    dry_run=True  # ← Automatically propagates to static_maker
)

# static_maker inherits dry_run=True automatically!
# No need to configure each child maker manually
```

**Supported flows** (all inherit from `BaseSiestaFlowMaker`):
- Convergence flows: `KpointsConvergenceMaker`, `MeshCutoffConvergenceMaker`, `BasisParametersConvergenceMaker`
- EOS flows: `EOSMaker`, `EOSBasisConvergenceMaker`
- Advanced workflows: `ElasticConstantsMaker`, `NEBMaker`
- Surface workflows: `SurfaceEnergyMaker`, `MultiSurfaceEnergyMaker`
- Adsorption workflows: `AdsorptionScanMaker`, `AdsorptionOptimizationMaker`

### 3. Job-Level vs Flow-Level Dry-Run

| Approach | Use Case | Example |
|----------|----------|---------|
| **Job-Level** | Single calculations | `RelaxMaker(..., dry_run=True)` |
| **Flow-Level** | Multi-step workflows | `KpointsConvergenceMaker(..., dry_run=True)` |

**Recommendation**: Use **flow-level** for most cases - it's simpler and ensures all child makers inherit dry-run mode automatically.

---

## Example 1: Job-Level Dry-Run (Basic)

### Simple Relaxation Preview

```python
from pymatgen.core import Structure
from jobflow import run_locally
from atomate2.siesta.jobs.core import RelaxMaker

# Load structure
structure = Structure.from_file("Si.cif")

# Create relax job with dry_run=True
relax_maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    dry_run_output_dir="preview_relax",
    dry_run_format="cif",
    dry_run_label="Si_test",
)

job = relax_maker.make(structure)
results = run_locally(job, create_folders=True)

# Output: preview_relax/Si_test/
#   - Si_test.cif
#   - siesta.fdf (inspect this!)
#   - structure.fdf
#   - Si.psml
```

### Key Parameters

```python
dry_run: bool = True                    # Enable dry-run mode
dry_run_output_dir: str = "preview"     # Base output directory
dry_run_format: str = "cif"             # Output format (cif, xsf, json, POSCAR)
dry_run_label: str = "job_name"         # Custom label for organization
```

### Inspecting Generated Files

```bash
# Check complete SIESTA input
cat preview_relax/Si_test/siesta.fdf

# Verify parameters
grep "PAO.BasisSize" preview_relax/Si_test/siesta.fdf
grep "kgrid" preview_relax/Si_test/siesta.fdf
grep "Mesh.Cutoff" preview_relax/Si_test/siesta.fdf

# Visualize structure
xcrysden --xsf preview_relax/Si_test/Si_test.xsf  # If XSF format
```

---

## Example 2: Flow-Level Dry-Run (Recommended)

### K-Points Convergence Study

Without dry-run:
```python
# ❌ This would run 5 full SIESTA calculations (~50 minutes)
flow = KpointsConvergenceMaker(
    kpoints_list=[[2,2,2], [4,4,4], [6,6,6], [8,8,8], [10,10,10]]
)
workflow = flow.make(structure)
results = run_locally(workflow)  # Runs all calculations!
```

With dry-run:
```python
# ✅ Preview all 5 calculations in ~5 seconds!
flow = KpointsConvergenceMaker(
    kpoints_list=[[2,2,2], [4,4,4], [6,6,6], [8,8,8], [10,10,10]],
    dry_run=True  # ← Just add this!
)
workflow = flow.make(structure)
results = run_locally(workflow)  # Generates structures only

# Inspect each k-point test:
# kpoints_preview/kpts_2_2_2/siesta.fdf
# kpoints_preview/kpts_4_4_4/siesta.fdf
# ...verify k-points are correct in each file
```

**Time saved**: 50 minutes → 5 seconds (99.8% reduction!)

### Mesh Cutoff Convergence Study

```python
from atomate2.siesta.flows.convergence import MeshCutoffConvergenceMaker

# Preview 7 mesh cutoff values
flow = MeshCutoffConvergenceMaker(
    mesh_cutoffs=[100, 150, 200, 250, 300, 350, 400],  # Ry
    dry_run=True,
    dry_run_output_dir="mesh_preview",
)

workflow = flow.make(structure)
results = run_locally(workflow)

# Generated: 7 complete SIESTA input packages
# mesh_preview/cutoff_100/siesta.fdf  # Check Mesh.Cutoff = 100 Ry
# mesh_preview/cutoff_150/siesta.fdf  # Check Mesh.Cutoff = 150 Ry
# ...
```

**Time saved**: 70 minutes → 5 seconds (99.9% reduction!)

### EOS Workflow

```python
from atomate2.siesta.flows.eos import EOSMaker

# Preview 7-point EOS calculation
flow = EOSMaker(
    linear_strain=(-0.05, 0.05),  # -5% to +5%
    number_of_frames=7,
    dry_run=True,
    dry_run_output_dir="eos_preview",
)

workflow = flow.make(structure)
results = run_locally(workflow)

# Generated: 7 volume points × complete input files
# eos_preview/strain_-0.050/siesta.fdf  # Check compressed structure
# eos_preview/strain_0.000/siesta.fdf   # Check equilibrium
# eos_preview/strain_0.050/siesta.fdf   # Check expanded structure
```

**Benefit**: Verify all 7 volume transformations before 70-minute calculation!

---

## Example 3: Different Output Formats

### Available Formats

```python
dry_run_format = "cif"      # Crystallographic Information File (DEFAULT)
dry_run_format = "xsf"      # XCrySDen format (visualization)
dry_run_format = "json"     # JSON format (programmatic analysis)
dry_run_format = "POSCAR"   # VASP format (compatibility)
```

### Format Comparison

| Format | Best For | Tool Compatibility |
|--------|----------|-------------------|
| **CIF** | General use, sharing | VESTA, Avogadro, Mercury |
| **XSF** | Visualization | XCrySDen, XCrySDen |
| **JSON** | Python scripts | pymatgen, custom analysis |
| **POSCAR** | VASP users | VESTA, p4vasp |

### Example: XSF for Visualization

```python
# Generate XSF files for XCrySDen visualization
maker = StaticMaker(
    dry_run=True,
    dry_run_format="xsf",
    dry_run_output_dir="viz_preview",
)

job = maker.make(structure)
results = run_locally(job)

# View in XCrySDen
# xcrysden --xsf viz_preview/*/structure.xsf
```

---

## Advanced Usage

### Custom Labels for Organization

Organize dry-run outputs with meaningful labels:

```python
# Test different basis sets
for basis in ["SZ", "DZ", "DZP", "DZDP"]:
    maker = RelaxMaker.fixed_cell_relaxation(
        dry_run=True,
        dry_run_label=f"Si_{basis}",
        user_params={"PAO.BasisSize": basis},
    )

    job = maker.make(structure)
    run_locally(job)

# Generated:
# preview_output/Si_SZ/siesta.fdf   # Check PAO.BasisSize SZ
# preview_output/Si_DZ/siesta.fdf   # Check PAO.BasisSize DZ
# preview_output/Si_DZP/siesta.fdf  # Check PAO.BasisSize DZP
# preview_output/Si_DZDP/siesta.fdf # Check PAO.BasisSize DZDP
```

### Workflow Summary Files

Dry-run automatically generates `workflow_summary.txt`:

```
WORKFLOW SUMMARY
================
Job Type: RelaxJob
Structure: Si2 (8.00 Ų)
Space Group: Fd-3m (227)

SIESTA Parameters:
------------------
PAO.BasisSize: DZP
Mesh.Cutoff: 300 Ry
k-points: 4 × 4 × 4
SCF.Mixer.Weight: 0.1
...

Generated Files:
----------------
✓ structure.cif
✓ siesta.fdf
✓ structure.fdf
✓ Si.psml
```

### Dry-Run in Complex Workflows

Dry-run works with nested workflows:

```python
from atomate2.siesta.flows.eos import EOSBasisConvergenceMaker

# Preview EOS + basis convergence study
# Tests 4 basis sets × 7 volumes = 28 calculations
flow = EOSBasisConvergenceMaker(
    basis_sets=["DZ", "DZP", "DZDP", "TZP"],
    linear_strain=(-0.05, 0.05),
    number_of_frames=7,
    dry_run=True,  # ← Propagates to ALL 28 child makers!
    dry_run_output_dir="eos_basis_preview",
)

workflow = flow.make(structure)
results = run_locally(workflow)

# Generated: 28 complete SIESTA input packages
# eos_basis_preview/DZ_strain_-0.05/siesta.fdf
# eos_basis_preview/DZ_strain_-0.04/siesta.fdf
# ...
# eos_basis_preview/TZP_strain_0.05/siesta.fdf

# Time saved: ~280 minutes → 10 seconds!
```

---

## Understanding Dry-Run Output

### What Gets Generated vs What Doesn't

✅ **Generated** (dry-run mode):
- Structure files (CIF, XSF, JSON, POSCAR)
- Complete `siesta.fdf` with ALL parameters
- `structure.fdf` with geometry
- Pseudopotential files (symlinks or copies)
- `siesta_parameters.json`
- `workflow_summary.txt`

❌ **NOT Generated** (dry-run mode):
- SIESTA calculation outputs (.out, .XV, .DM, .EIG, etc.)
- Energy values
- Forces/stresses
- Electronic structure data
- Convergence information

### Transition from Dry-Run to Real Calculation

Once you've validated the workflow:

```python
# Step 1: Dry-run to validate
flow = KpointsConvergenceMaker(..., dry_run=True)
workflow = flow.make(structure)
run_locally(workflow)

# Step 2: Inspect generated siesta.fdf files
# cat kpoints_preview/*/siesta.fdf

# Step 3: Run actual calculations (remove dry_run)
flow = KpointsConvergenceMaker(...)  # ← No dry_run parameter
workflow = flow.make(structure)
run_locally(workflow)  # Runs real SIESTA calculations
```

---

## Best Practices

### 1. Always Dry-Run First

**Before running**:
- Any convergence study (k-points, cutoff, basis)
- Expensive workflows (EOS, elastic constants, phonons)
- New workflow types you haven't used before
- Complex multi-step workflows

**Why**: Catch configuration errors before wasting computational time!

### 2. Inspect siesta.fdf Files

Critical checks:
```bash
# Verify basis set
grep "PAO.BasisSize" preview_output/*/siesta.fdf

# Verify k-points
grep "kgrid" preview_output/*/siesta.fdf

# Verify mesh cutoff
grep "Mesh.Cutoff" preview_output/*/siesta.fdf

# Verify SCF settings
grep "SCF" preview_output/*/siesta.fdf

# Check all parameters
cat preview_output/job_name/siesta.fdf | less
```

### 3. Use Flow-Level Dry-Run

**Recommended**:
```python
flow = KpointsConvergenceMaker(..., dry_run=True)  # ✅ Set once
```

**Avoid (manual configuration)**:
```python
static_maker = StaticMaker(dry_run=True)  # ❌ Must configure each maker
flow = KpointsConvergenceMaker(static_maker=static_maker)
```

### 4. Organize with Custom Labels

Use descriptive labels for easy navigation:
```python
dry_run_label = f"{formula}_{parameter}_{value}"
# Example: "Si_kpts_4x4x4", "GaAs_cutoff_300Ry"
```

### 5. Choose Appropriate Format

- **Default (CIF)**: Good for most cases
- **XSF**: If you'll visualize with XCrySDen
- **JSON**: If you'll parse structures with Python
- **POSCAR**: If collaborating with VASP users

---

## Troubleshooting

### Issue 1: Dry-Run Not Generating Files

**Problem**: No output files generated

**Solution**:
```python
# Make sure dry_run=True is set
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)  # ← Check this

# Verify output directory exists or will be created
dry_run_output_dir = "preview"  # Will be created automatically

# Check run_locally has create_folders=True
results = run_locally(job, create_folders=True)  # ← Important!
```

### Issue 2: siesta.fdf Missing Parameters

**Problem**: Expected parameters not in `siesta.fdf`

**Solution**:
```python
# Parameters must be set in user_params
maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    user_params={
        "PAO.BasisSize": "DZP",        # ← Set explicitly
        "Mesh.Cutoff": "300 Ry",
        "kpts": [6, 6, 6],
    }
)
```

### Issue 3: Flow-Level Dry-Run Not Propagating

**Problem**: Child makers not inheriting dry-run

**Solution**:
```python
# Ensure flow inherits from BaseSiestaFlowMaker
# (All built-in flows do this automatically)

# Set dry_run at flow level, NOT maker level
flow = KpointsConvergenceMaker(
    ...,
    dry_run=True  # ← Here (flow level)
)

# DON'T do this:
static_maker = StaticMaker(dry_run=True)  # ← Avoid manual config
flow = KpointsConvergenceMaker(static_maker=static_maker)
```

### Issue 4: Pseudopotential Files Missing

**Problem**: `.psml` or `.psf` files not in output

**Solution**:
```python
# Set SIESTA_PP_PATH environment variable
import os
os.environ["SIESTA_PP_PATH"] = "/path/to/pseudopotentials"

# Or in ~/.atomate2siesta.yaml:
# SIESTA_PP_PATH: /path/to/pseudopotentials
```

### Issue 5: Can't Find Generated Files

**Problem**: Don't know where files were saved

**Solution**:
```python
# Check dry_run_output_dir and dry_run_label
dry_run_output_dir = "my_preview"  # Base directory
dry_run_label = "Si_test"          # Subdirectory

# Files will be in: my_preview/Si_test/

# Default locations:
# - Job-level: preview_output/<job_uuid>/
# - Flow-level: <dry_run_output_dir>/<parameter_value>/
```

---

## Configuration Options

### Complete Dry-Run Parameters

```python
maker = RelaxMaker.fixed_cell_relaxation(
    # Dry-run control
    dry_run: bool = True,                    # Enable dry-run mode

    # Output configuration
    dry_run_output_dir: str = "preview",     # Base directory
    dry_run_label: str = "job_name",         # Subdirectory name
    dry_run_format: str = "cif",             # Output format

    # Calculation parameters (still used in siesta.fdf generation)
    user_params: dict = {
        "PAO.BasisSize": "DZP",
        "Mesh.Cutoff": "300 Ry",
        "kpts": [4, 4, 4],
    }
)
```

### Format Options

```python
dry_run_format = "cif"      # Crystallographic Information File
dry_run_format = "xsf"      # XCrySDen format
dry_run_format = "json"     # JSON format
dry_run_format = "POSCAR"   # VASP POSCAR format
dry_run_format = "yaml"     # YAML format
dry_run_format = "xdatcar"  # VASP XDATCAR format (trajectories)
```

---

## Integration with Other Features

### With Powerups

```python
from atomate2.siesta.powerups import update_user_siesta_settings

# Create flow with dry-run
flow = KpointsConvergenceMaker(..., dry_run=True)
workflow = flow.make(structure)

# Apply powerups
workflow = update_user_siesta_settings(
    workflow,
    {"Mesh.Cutoff": "400 Ry"}
)

# Dry-run will include updated parameters
run_locally(workflow)
```

### With Tier System

```python
from atomate2.siesta.sets.tiers import apply_tier_preset

# Create maker with preset
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(maker, "relax_standard")

# Dry-run includes all tier parameters
job = maker.make(structure)
run_locally(job)
```

### With Custodian

```python
# Dry-run ignores custodian settings
maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    use_custodian=True,  # ← Ignored in dry-run mode
)

# No error handling needed - no calculations run!
```

---

## Performance Comparison

### Typical Time Savings

| Workflow | Real Calculation | Dry-Run | Time Saved |
|----------|-----------------|---------|------------|
| Single relax | 10 min | 1 sec | 99.8% |
| K-points (5 points) | 50 min | 3 sec | 99.9% |
| Mesh cutoff (7 points) | 70 min | 5 sec | 99.9% |
| EOS (7 volumes) | 70 min | 5 sec | 99.9% |
| EOS + basis (28 calcs) | 280 min | 10 sec | 99.9% |
| Phonon (50 displacements) | 500 min | 15 sec | 99.9% |

### Storage Requirements

Dry-run output is minimal:
- Structure file: ~1-10 KB
- siesta.fdf: ~5-10 KB
- structure.fdf: ~1-5 KB
- Total per job: ~10-50 KB

vs. Real calculation output: ~1-100 MB per job

---

## Next Steps

1. **Try the tutorial**:
   ```bash
   cd tutorials/04-infrastructure/05-dry-run-preview
   python tutorial.py
   ```

2. **Experiment with example types**:
   - `EXAMPLE_TYPE = "job_level"` - Basic dry-run
   - `EXAMPLE_TYPE = "flow_level"` - Workflow dry-run (recommended)
   - `EXAMPLE_TYPE = "formats"` - Test output formats

3. **Inspect generated files**:
   ```bash
   cat preview_output/*/siesta.fdf
   xcrysden --xsf preview_output/*/*.xsf
   ```

4. **Apply to your own workflows**:
   - Add `dry_run=True` to any maker or flow
   - Validate before running expensive calculations
   - Save hours of computational time!

---

## Summary

### Key Takeaways

✅ **Dry-run mode** generates complete SIESTA input files without running calculations
✅ **Time savings**: 99.9% reduction in validation time (hours → seconds)
✅ **Automatic propagation**: Set `dry_run=True` once at flow level
✅ **Zero cost**: Perfect for learning and validation
✅ **Universal support**: Works with all makers and flows

### When to Use

🎯 **Always** before expensive workflows
🎯 **Always** for convergence studies
🎯 **Always** when learning new features
🎯 **Always** to inspect SIESTA parameters

### Quick Reference

```python
# Job-level (single calculation)
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)

# Flow-level (multi-step workflow) - RECOMMENDED
flow = KpointsConvergenceMaker(..., dry_run=True)

# Inspect generated parameters
cat preview_output/*/siesta.fdf
```

**Make dry-run your first step for every new workflow!** 🚀
