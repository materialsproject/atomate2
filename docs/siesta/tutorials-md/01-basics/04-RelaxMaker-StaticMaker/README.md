# Tutorial: Multi-Step Workflows

**Category**: 01-basics
**Difficulty**: Beginner
**Time**: ~5 min (dry-run), ~30-45 min (full calculation)

---

## Overview

Learn how to create multi-step workflows by chaining multiple jobs together. This tutorial demonstrates using Flow makers to automatically run calculations with different parameters and compare results.

---

## What You'll Learn

- Creating multi-job workflows with Flow makers
- `DifferentBasisRelaxMaker` for basis set comparison
- Job dependencies and data flow
- Automatic result comparison
- Applying powerups to entire workflows
- Sequential vs parallel job execution

---

## Prerequisites

- **Required**: [01-relaxation](../01-RelaxMaker/) completed
- **Required**: [02-relaxation-parameters](../02-BandStructureMaker/)
- **Recommended**: Understanding of SIESTA basis sets

---

## Key Concepts

### What is a Workflow (Flow)?

A **workflow** (or **flow**) is a collection of jobs with dependencies. Jobs can run:

- **Sequentially**: Job B waits for Job A to finish
- **In parallel**: Independent jobs run simultaneously
- **Conditionally**: Job C runs only if Job B succeeds

**Single Job**:
```python
job = RelaxMaker().make(structure)
# One calculation
```

**Workflow (Flow)**:
```python
flow = DifferentBasisRelaxMaker().make(structure)
# Multiple related calculations
```

### Restart Techniques (NEW!)

SIESTA provides powerful restart capabilities for faster calculations:

**Density Matrix (DM) Restart**:
- **What**: Reuse converged electronic structure from previous calculation
- **Benefit**: 50-70% fewer SCF iterations
- **Use when**: Refining k-points, tightening tolerance, adding post-processing
- **Tutorial**: `02_restart_from_dm.py`

**Geometry (XV) Restart**:
- **What**: Continue geometry optimization from previous atomic positions
- **Benefit**: 50-70% fewer relaxation steps
- **Use when**: Two-stage relaxation (loose→tight), resuming interrupted jobs
- **Tutorial**: `03_restart_from_xv.py`

**Combined DM + XV**:
- Maximum efficiency: faster SCF AND fewer geometry steps
- Ideal for multi-stage high-accuracy workflows

---

## Restart Techniques: Detailed Guide

### Density Matrix (DM) Restart

**Physics Background**:

The density matrix contains the self-consistent electronic structure solution. By starting with a converged DM from a similar calculation, SIESTA can:
- Skip many SCF iterations (typically 50-70% reduction)
- Achieve better initial guess for difficult-to-converge systems
- Maintain consistent orbital character across parameter changes

**When to Use**:
- ✓ Refining k-point mesh (2×2×2 → 4×4×4)
- ✓ Tightening SCF convergence criteria
- ✓ Small geometry adjustments
- ✓ Adding post-processing (DOS, bands) to converged structure
- ✗ Large structural changes (DM won't match well)
- ✗ Changing number of atoms (DM incompatible)
- ✗ Changing basis set (orbital count changes)

**Key Parameters**:
```python
user_params={
    "DM.UseSaveDM": True,  # MUST be true in both jobs
    "PAO.BasisSize": "DZ",  # MUST be identical between jobs
}
```

**File Dependency**:
```python
# Job 2 must use prev_dir to access DM file from Job 1
fine_job = StaticMaker(...).make(structure, prev_dir=coarse_job.output.dir_name)
```

**Expected Performance**:
- Coarse job: ~10-15 SCF iterations
- Fine job with DM: ~3-5 SCF iterations (vs 10-15 without)
- Total time saving: 30-50%

---

### Geometry (XV) Restart

**Physics Background**:

The XV file contains atomic positions and velocities from the last geometry optimization step. By continuing from a partially optimized structure:
- Skip redundant relaxation steps (typically 50-70% reduction)
- Maintain optimization momentum (CG history)
- Reach tight convergence faster

**When to Use**:
- ✓ Two-stage relaxation (loose → tight convergence criteria)
- ✓ Interrupted calculations (power loss, time limit exceeded)
- ✓ Refining pre-optimized geometry with better parameters
- ✓ Continuing MD simulations with velocities
- ✗ Starting from unoptimized structure (no benefit)
- ✗ Large parameter changes requiring re-optimization

**Key Parameters**:
```python
user_params={
    "MD.UseSaveXV": True,  # MUST be true in both jobs
    "MD.MaxForceTol": "0.04 eV/Ang",  # Job 1: loose
    "MD.MaxForceTol": "0.01 eV/Ang",  # Job 2: tight
}
```

**File Dependency**:
```python
# Job 2 must use prev_dir to access XV file from Job 1
fine_relax = RelaxMaker(...).make(structure, prev_dir=quick_relax.output.dir_name)
```

**Expected Performance**:
- Quick relax: ~15-20 steps to 0.04 eV/Å
- Fine relax with XV: ~5-10 steps to 0.01 eV/Å (vs 25-35 from scratch)
- Total time saving: 40-60%

---

## Analysis Techniques

### DM Restart Analysis

**1. Compare SCF Iterations**:
```bash
# Count SCF cycles in each job
for job in dm_restart_workflow/job_*/; do
    count=$(grep -c "scf:" $job/siesta.out)
    echo "$(basename $job): $count iterations"
done
```

**2. Verify DM File Was Read**:
```bash
grep "Reading density-matrix from file" dm_restart_workflow/job_*/siesta.out
```

**3. Compare Final Energies**:
```bash
grep "siesta: E_KS(eV)" dm_restart_workflow/job_*/siesta.out | tail -2
# Should be nearly identical (difference < 1 meV)
```

**4. Check DM File Sizes**:
```bash
ls -lh dm_restart_workflow/job_*/*.DM
# Larger DM files = more basis functions
```

---

### XV Restart Analysis

**1. Compare Total Relaxation Steps**:
```bash
for job in xv_restart_workflow/job_*/; do
    grep "Relaxed in" $job/siesta.out
done
# Example output:
# Job 1: Relaxed in 18 steps
# Job 2: Relaxed in 7 steps
# Total: 25 steps (vs ~35 from scratch = 30% savings)
```

**2. Verify XV File Was Read**:
```bash
grep "Reading coordinates and velocities" xv_restart_workflow/job_*/siesta.out
```

**3. Compare Final Structures**:
```python
from pymatgen.core import Structure
s1 = Structure.from_file('xv_restart_workflow/job_*_1/siesta.XV')
s2 = Structure.from_file('xv_restart_workflow/job_*_2/siesta.XV')
print(f"Volume change Job1→Job2: {abs(s2.volume - s1.volume):.4f} Å³")
print(f"Max atomic displacement: {max_displacement(s1, s2):.4f} Å")
```

**4. Track Force Convergence**:
```python
import re
for job_dir in ['job_*_1', 'job_*_2']:
    with open(f'{job_dir}/siesta.out') as f:
        forces = re.findall(r'siesta: Atomic forces.*?:\s+([-\d.]+)', f.read())
    print(f"{job_dir}:")
    print(f"  Initial max force: {forces[0]} eV/Å")
    print(f"  Final max force: {forces[-1]} eV/Å")
```

---

## Best Practices for Restart Workflows

**DM Restart**:
1. **Identical basis**: PAO.BasisSize must match exactly
2. **Compatible geometry**: Similar atomic positions (< 0.5 Å displacement)
3. **Progressive refinement**: Gradually increase k-points, tighten tolerance
4. **Check convergence**: Verify energies match between jobs (< 1 meV)

**XV Restart**:
1. **Sequential tolerances**: Loose → Medium → Tight (e.g., 0.1 → 0.04 → 0.01 eV/Å)
2. **Consistent parameters**: Keep basis, k-points, mesh cutoff same
3. **Monitor progress**: Check if second stage needs fewer steps
4. **Appropriate step limits**: MD.NumCGsteps should allow convergence

**Combined DM + XV**:
1. Use for maximum efficiency in multi-stage workflows
2. Example: Quick relax (saves XV+DM) → Fine relax (reads both)
3. Ideal for high-accuracy production calculations

---

### Benefits of Workflows

✓ **Automatic job chaining**: No manual setup between steps
✓ **Data flow between jobs**: Results automatically passed
✓ **Efficient resource usage**: Parallel execution when possible
✓ **Reproducible calculations**: Entire workflow defined in code
✓ **Parameter studies**: Test multiple settings systematically

### Common Workflow Types

**Convergence Testing**:
- Test multiple k-point meshes
- Vary mesh cutoff values
- Compare basis sets (this tutorial)

**Sequential Refinement**:
- Coarse calculation → Fine calculation
- Quick relax → Accurate relax
- Static → Band structure

**Property Mapping**:
- Same structure, different properties
- Multiple structures, same property
- High-throughput screening

**Multi-step Calculations**:
- Relax → DOS
- Relax → Band structure → DOS
- Relax → Phonons → Thermal properties

---

## DifferentBasisRelaxMaker

This tutorial uses `DifferentBasisRelaxMaker` which automatically creates multiple relaxation jobs with different basis sets.

**Default Basis Sets Tested**:
1. **SZ** (Single-Zeta) - Minimal basis, fastest
2. **DZ** (Double-Zeta) - Standard basis
3. **DZP** (Double-Zeta + Polarization) - High accuracy

**Purpose**: Determine minimum basis set needed for converged results.

```python
from atomate2.siesta.flows.core import DifferentBasisRelaxMaker

# Creates 3 jobs automatically
workflow = DifferentBasisRelaxMaker().make(structure)

# Apply common parameters to all jobs
workflow = update_user_siesta_settings(workflow, {
    "kpts": [4, 4, 4],
    "Mesh.Cutoff": "200 Ry",
})
```

---

## Configuration Options

### Workflow Creation

The tutorial uses default settings (SZ, DZ, DZP), but you can customize:

```python
# Default (in this tutorial)
workflow_maker = DifferentBasisRelaxMaker()

# Custom basis sets
workflow_maker = DifferentBasisRelaxMaker(
    basis_sizes=["DZ", "DZP", "TZP"]  # Skip SZ, add TZP
)
```

### Common Parameters

Parameters can be applied to all jobs in the workflow:

```python
workflow = update_user_siesta_settings(workflow, {
    "Mesh.Cutoff": "200 Ry",  # Applied to all 3 jobs
    "kpts": [4, 4, 4],        # Applied to all 3 jobs
})
```

---

## Tutorial Files

This directory contains three workflow tutorials:

1. **`01_sequential.py`** - Basic multi-job workflow (basis set comparison)
2. **`02_restart_from_dm.py`** - Using previous density matrix for faster SCF ⭐ NEW!
3. **`03_restart_from_xv.py`** - Using previous geometry for continued relaxation ⭐ NEW!

## Quick Start

### Tutorial 1: Sequential Workflow (Basis Comparison)

```bash
# 1. Preview workflow structure
python 01_sequential.py  # RUN_MODE = "dry_run"

# 2. Inspect job folders
ls preview_output/job_*
# Should see: job_*_SZ/, job_*_DZ/, job_*_DZP/

# 3. Check each has different basis
grep "PAO.BasisSize" preview_output/job_*/siesta.fdf

# 4. Run workflow (3 jobs sequentially or parallel)
# Edit: RUN_MODE = "local"
python 01_sequential.py
```

### Tutorial 2: DM Restart (Fast SCF Convergence)

```bash
# 1. Preview two-stage workflow
python 02_restart_from_dm.py  # RUN_MODE = "dry_run"

# 2. Check DM restart settings
grep "DM.UseSaveDM" dm_restart_workflow/job_*/siesta.fdf
grep "kpts" dm_restart_workflow/job_*/siesta.fdf

# 3. Run workflow
# Edit: RUN_MODE = "local"
python 02_restart_from_dm.py

# 4. Compare SCF iterations (should be 50-70% fewer in Job 2)
grep "scf:" dm_restart_workflow/job_*/siesta.out | wc -l
```

### Tutorial 3: XV Restart (Geometry Continuation)

```bash
# 1. Preview two-stage relaxation
python 03_restart_from_xv.py  # RUN_MODE = "dry_run"

# 2. Check relaxation settings
grep "MD.MaxForceTol" xv_restart_workflow/job_*/siesta.fdf
# Job 1: 0.04 eV/Ang (loose)
# Job 2: 0.01 eV/Ang (tight)

# 3. Run workflow
# Edit: RUN_MODE = "local"
python 03_restart_from_xv.py

# 4. Compare relaxation steps (total should be 50-70% less)
grep "Relaxed in" xv_restart_workflow/job_*/siesta.out
```

---

## Expected Output

### Dry-Run Mode

```
✅ Workflow created with 3 jobs
   Each job uses a different basis set

🔍 DRY-RUN MODE
─────────────────────────────────────────────
Previewing workflow structure
This workflow contains multiple jobs:
─────────────────────────────────────────────

✅ Dry-run complete!

📁 Created job folders:
  job_*_SZ/  - Single-zeta calculation
  job_*_DZ/  - Double-zeta calculation
  job_*_DZP/ - Double-zeta + polarization

💡 Each folder has complete SIESTA input files
💡 Jobs will run in sequence (or parallel if independent)
```

**Directory Structure**:
```
preview_output/
├── job_2024XXXX_XXXXXX_SZ/
│   ├── siesta.fdf          # PAO.BasisSize SZ
│   ├── structure.fdf
│   └── ...
├── job_2024XXXX_XXXXXX_DZ/
│   ├── siesta.fdf          # PAO.BasisSize DZ
│   └── ...
└── job_2024XXXX_XXXXXX_DZP/
    ├── siesta.fdf          # PAO.BasisSize DZP
    └── ...
```

### Local Mode

```
▶️  LOCAL EXECUTION
─────────────────────────────────────────────
Running 3 relaxation calculations sequentially
Total time: ~30-45 minutes (3 × 10-15 min)
─────────────────────────────────────────────

✅ All jobs complete!

📊 Results:
  Check job_* folders for individual results
  Compare final energies across basis sets
```

---

## Analyzing Results

### Extract Final Energies

```bash
# Method 1: Direct from SIESTA output
grep "siesta: E_KS(eV)" job_*/siesta.out

# Example output:
# job_*_SZ/siesta.out:  siesta: E_KS(eV) = -31.456789
# job_*_DZ/siesta.out:  siesta: E_KS(eV) = -31.523456
# job_*_DZP/siesta.out: siesta: E_KS(eV) = -31.534567

# Method 2: Using Python
python -c "
from pymatgen.io.siesta import SiestaOutput
import glob

for folder in sorted(glob.glob('job_*')):
    output = SiestaOutput(f'{folder}/siesta.out')
    basis = folder.split('_')[-1]
    print(f'{basis:5s}: {output.final_energy:.6f} eV')
"
```

### Compare Structures

```python
# Compare final volumes
from pymatgen.core import Structure
import glob

for folder in sorted(glob.glob('job_*')):
    structure = Structure.from_file(f'{folder}/structure.cif')
    basis = folder.split('_')[-1]
    print(f"{basis}: {structure.volume:.3f} Å³")
```

### Convergence Analysis

**Good Convergence**:
```
SZ:  -31.456789 eV  (Δ = 0.078 eV from DZP)
DZ:  -31.523456 eV  (Δ = 0.011 eV from DZP)
DZP: -31.534567 eV  (reference)
```
→ **Conclusion**: DZ sufficient (< 20 meV/atom difference)

**Poor Convergence**:
```
SZ:  -31.456789 eV  (Δ = 0.178 eV from DZP)
DZ:  -31.523456 eV  (Δ = 0.111 eV from DZP)
DZP: -31.634567 eV  (reference)
```
→ **Conclusion**: Need DZP or higher

### Decision Criteria

**Energy Difference Thresholds**:
- < 10 meV/atom: Excellent convergence
- 10-25 meV/atom: Good convergence (acceptable for most purposes)
- 25-50 meV/atom: Marginal convergence (use with caution)
- > 50 meV/atom: Poor convergence (use higher basis)

---

## Common Issues

### Issue 1: "Jobs running too long"

**Symptoms**: Workflow takes hours to complete

**Solutions**:
1. **Run in parallel**: If resources allow
   - Requires HPC cluster setup (see [04-infrastructure](../../../03-advanced-features/03-infrastructure/))
   - Use `RUN_MODE = "submit"` with jobflow-remote

2. **Reduce system size**: Test with smaller structure first

3. **Use dry-run**: Verify setup before running

### Issue 2: "Can't compare results"

**Symptoms**: Different job folders have different structures

**Solution**: Ensure all jobs used same initial structure
```bash
# Check initial structures are identical
md5sum job_*/structure.fdf
# All should have same hash
```

### Issue 3: "Basis sets not applied"

**Symptoms**: All jobs use same basis

**Solution**: Check powerups didn't override basis settings
```python
# ❌ WRONG - This overrides workflow basis settings
workflow = DifferentBasisRelaxMaker().make(structure)
workflow = update_user_siesta_settings(workflow, {
    "PAO.BasisSize": "DZP"  # This overrides all jobs!
})

# ✓ CORRECT - Let workflow control basis
workflow = DifferentBasisRelaxMaker().make(structure)
workflow = update_user_siesta_settings(workflow, {
    "kpts": [4, 4, 4]  # Other parameters OK
})
```

### Issue 4: "Workflow stopped after first job"

**Symptoms**: Only job_*_SZ folder has results

**Possible Causes**:
1. **First job failed**: Check siesta.out for errors
2. **Resource limits**: Ran out of time/memory
3. **Dependency issue**: Jobs waiting for failed job

**Debug**:
```bash
# Check all job logs
tail -20 job_*/siesta.out

# Look for errors
grep -i "error\|abort" job_*/siesta.out
```

### Issue 5: "Unknown FDF parameter: fdf_arguments"

**Error**: `ValueError: Unknown FDF parameter(s): fdf_arguments`

**Cause**: Using deprecated `fdf_arguments` wrapper syntax

**Solution**:
```python
# ❌ OLD (doesn't work):
user_params = {
    "fdf_arguments": {
        "DM.InitSpin": [...]
    }
}

# ✅ NEW (correct):
user_params = {
    "%block DM.InitSpin": [...]  # Directly in user_params!
}
```

**Note**: Block parameters like `%block DM.InitSpin`, `%block Geometry.Constraints`, and `%block DFTU.Proj` should be specified **directly** in `user_params`, NOT nested in `fdf_arguments`.

---

## Available Flow Makers

atomate2siesta provides many workflow types:

### Convergence Studies

**DifferentBasisRelaxMaker**:
- Compare basis sets (SZ, DZ, DZP, TZP)
- This tutorial

**ConvergenceMaker** (see [02-convergence](../../../02-workflows/01-convergence/)):
- K-points convergence
- Mesh cutoff convergence
- Systematic parameter testing

### Property Calculations

**BandStructureWorkflow**:
- Relax → Static → Band structure
- Automatic high-symmetry path

**PhononMaker** (see [16-phonon-calculations](../../../02-workflows/06-vibrational-properties/)):
- Relax → Force constants → Phonon bands
- Thermal properties calculation

**EOSMaker** (see [09-equation-of-state](../../../02-workflows/01-equation-of-state/)):
- Multiple volumes → Fit equation of state
- Bulk modulus, equilibrium volume

### Advanced Workflows

**NudgedElasticBandMaker**:
- Reaction path calculations
- Transition state search

**ElasticConstantsMaker**:
- Multiple deformations → Elastic tensor
- Mechanical properties

**SurfaceEnergyMaker** (see [17-surface-energy](../../../02-workflows/03-surfaces-and-adsorption/01-surface-energy/)):
- Multiple surface orientations
- Convergence vs slab thickness

---

## Workflow Execution Modes

### Sequential Execution

Jobs run one after another:
```
Job 1 (SZ) → Job 2 (DZ) → Job 3 (DZP)
Total time = Sum of all jobs
```

**When**: Using `run_locally` (local mode)

### Parallel Execution

Independent jobs run simultaneously:
```
Job 1 (SZ)  ┐
Job 2 (DZ)  ├─ All run at once
Job 3 (DZP) ┘
Total time ≈ Longest job
```

**When**: Using jobflow-remote with cluster (submit mode)

### Hybrid (Sequential + Parallel)

Mixed dependencies:
```
Job 1 (Relax) → Job 2a (Band structure) ┐
                Job 2b (DOS)           ├─ Parallel
                Job 2c (Phonons)       ┘
```

**When**: Complex workflows with dependencies

---

## Creating Custom Workflows

### Basic Pattern

```python
from jobflow import Flow
from atomate2.siesta.jobs.core import RelaxMaker

# Create individual jobs
relax_job = RelaxMaker().make(structure)
dos_job = DOSMaker().make(structure, prev_dir=relax_job.output)

# Combine into workflow
workflow = Flow([relax_job, dos_job])
```

### Workflow with Multiple Parameters

```python
# Test multiple k-point meshes
kpoints_list = [[2,2,2], [4,4,4], [6,6,6], [8,8,8]]
jobs = []

for kpts in kpoints_list:
    job = RelaxMaker().make(structure)
    job = update_user_siesta_settings(job, {"kpts": kpts})
    jobs.append(job)

workflow = Flow(jobs)
```

---

## Next Steps

After completing this tutorial:

1. **Convergence studies**: [02-convergence](../../../02-workflows/01-convergence/) category
2. **Advanced workflows**: [03-advanced-workflows](../../../02-workflows/) category
3. **HPC submission**: [04-infrastructure/02-job-submission](../../../03-advanced-features/03-infrastructure/02-job-submission/)
4. **Custom workflows**: Learn Flow and jobflow patterns

---

## FDF Block Parameters (Advanced)

When using workflows with custom FDF block parameters, use the `"%block ParamName"` syntax **directly** in `user_params`.

**IMPORTANT**: DO NOT wrap block parameters in `fdf_arguments` - this is deprecated!

### Correct Usage in Workflows

```python
# ✅ CORRECT: Block parameters directly in user_params
from atomate2.siesta.powerups import update_user_siesta_settings

workflow = DifferentBasisRelaxMaker().make(structure)
workflow = update_user_siesta_settings(
    workflow,
    {
        "a2s_kpts": [6, 6, 6],
        "Spin": "polarized",

        # DM.InitSpin block for magnetic systems
        "%block DM.InitSpin": [
            "1  +2.0",
            "2  -2.0",
        ],
    },
)
```

### Incorrect Usage (Deprecated)

```python
# ❌ WRONG: Don't nest in fdf_arguments!
workflow = update_user_siesta_settings(
    workflow,
    {
        "fdf_arguments": {  # <-- This doesn't work!
            "DM.InitSpin": [...]
        }
    },
)
```

**Common block parameters for workflows**:
- `"%block DM.InitSpin"` - Initial magnetic moments
- `"%block Geometry.Constraints"` - Fix atoms during relaxation
- `"%block DFTU.Proj"` - DFT+U projectors

For comprehensive examples, see [02-fdf-block-inputs](../../03-advanced-features/02-fdf-block-inputs/).

---

## Tips for Success

✅ **Start with dry-run**: Verify workflow structure before running
✅ **Apply common parameters carefully**: Don't override workflow-specific settings
✅ **Use descriptive job names**: Helps identify jobs in complex workflows
✅ **Check job dependencies**: Ensure jobs run in correct order
✅ **Monitor parallel jobs**: Watch for resource conflicts on shared systems
✅ **Save results systematically**: Use clear naming conventions
✅ **Block parameters**: Use `"%block ParamName"` directly in `user_params` - NO `fdf_arguments` wrapper!

---

## Best Practices

**Workflow Design**:
1. **Modularity**: Break complex tasks into simple jobs
2. **Reusability**: Use makers instead of custom functions
3. **Clarity**: Name jobs descriptively
4. **Testing**: Test workflow with small system first

**Parameter Studies**:
1. **One variable at a time**: Easier to analyze
2. **Systematic ranges**: Linear or logarithmic spacing
3. **Document choices**: Why these parameters?
4. **Plot results**: Energy vs parameter curves

**Resource Management**:
1. **Estimate time**: How long will workflow take?
2. **Check disk space**: Multiple jobs = many files
3. **Parallel wisely**: Don't overload system
4. **Clean up**: Remove intermediate files

---

*Back to [01-basics](../README.md) | [Main Tutorial Index](../../README.md)*
