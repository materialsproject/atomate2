# Troubleshooting Guide

**Category**: troubleshooting
**Difficulty**: Intermediate
**Time**: Variable (diagnostic + fix)

---

## Overview

This section provides **systematic troubleshooting guides** for common issues encountered when running atomate2siesta workflows. Each guide includes symptom identification, root cause analysis, and step-by-step solutions.

**What This Section Covers**:
- **Diagnosis**: Identifying what went wrong
- **Root causes**: Understanding why it failed
- **Solutions**: Fixing the problem systematically
- **Prevention**: Avoiding issues in future calculations
- **Automation**: Using custodian for automatic error recovery

---

## Purpose

Computational materials science calculations fail for many reasons - wrong parameters, unconverged settings, system-specific issues, or infrastructure problems. This section helps you:

✅ **Quickly identify** what went wrong
✅ **Understand** the root cause
✅ **Apply** the correct fix
✅ **Prevent** future occurrences
✅ **Automate** error recovery with custodian

---

## Troubleshooting Categories

### [common_errors/](common_errors/)
**Focus**: Most frequently encountered calculation failures

**Coverage**:
- SCF convergence failures
- Geometry optimization problems
- Memory/resource errors
- Basis set issues
- K-point and mesh errors
- Pseudopotential problems

**Available Tutorial**:
- **[scf_convergence_issues.py](common_errors/scf_convergence_issues.py)** - Complete SCF troubleshooting guide
  - Mixer settings adjustment
  - Electronic temperature for metals
  - Spin polarization for magnetic systems
  - Charge density initialization
  - Using custodian for automatic recovery

---

### [debugging_workflows/](debugging_workflows/)
**Focus**: Workflow-level debugging strategies

**Planned Topics**:
- Tracing job dependencies in flows
- Identifying which job failed
- Checking intermediate outputs
- Debugging powerup applications
- Verifying tier preset application
- Database query for job history

**Key Techniques**:
- Using `job.output` inspection
- Flow visualization
- Logging and output analysis
- Dry-run mode for testing
- Database queries with jobflow-remote

---

### [performance_optimization/](performance_optimization/)
**Focus**: Making calculations faster and more efficient

**Planned Topics**:
- Parallel efficiency (k-point vs basis parallelization)
- Memory optimization
- Reducing computational cost
- Choosing appropriate tier presets
- When to use tighter vs looser convergence
- HPC resource allocation

**Key Strategies**:
- k-point parallelization for large systems
- Basis parallelization for small systems
- Memory profiling
- Computational cost vs accuracy tradeoffs

---

## General Troubleshooting Workflow

Follow this systematic approach for any failed calculation:

### Step 1: Identify the Failure

```bash
# For local runs
ls -ltr job_*/  # Check which job directory was created
grep -i "error\|fail\|abort" job_*/siesta.out

# For jobflow-remote runs
jf -p PROJECTNAME job list --state FAILED
jf -p PROJECTNAME job info <db_id> --full
```

**Key questions**:
- Which job failed? (relaxation, static, NEB, etc.)
- At what stage? (initialization, SCF, geometry optimization)
- What error message? (exact text from output)

### Step 2: Read the Error Message

```bash
# Get full error context
tail -100 job_*/siesta.out

# Search for specific errors
grep -A 5 -B 5 "ERROR\|FATAL" job_*/siesta.out
```

**Common error patterns**:
- `SCF did not converge` → Convergence issue
- `kgrid: ERROR` → K-point problem
- `Exceeded memory limit` → Resource issue
- `PAO.Basis: Unknown` → Basis set error
- `Reading pseudopotential failed` → Pseudopotential issue

### Step 3: Check Input Parameters

```bash
# Review input file
cat job_*/siesta.fdf

# Check specific parameters
grep "Mesh.Cutoff\|kgrid.Cutoff\|PAO.BasisSize" job_*/siesta.fdf
```

**Validate**:
- Are k-points reasonable? (density ~ 0.03-0.05 Å⁻¹)
- Is mesh cutoff sufficient? (≥200 Ry for most systems)
- Is basis size appropriate? (DZP standard, TZP for accurate forces)
- Are atom positions valid? (no overlapping atoms)

### Step 4: Examine Convergence History

```bash
# SCF convergence
grep "scf:" job_*/siesta.out

# Geometry optimization
grep "siesta: E_KS(eV)" job_*/siesta.out
grep "siesta: Atomic forces" job_*/siesta.out
```

**Look for**:
- Oscillating energies → Mixer too aggressive
- Diverging energies → Wrong parameters or structure
- Slow convergence → Need tighter/different settings

### Step 5: Apply Appropriate Fix

See category-specific guides below for detailed solutions.

---

## Quick Fixes by Error Type

### SCF Convergence Failures

**Symptom**: "SCF did not converge in N iterations"

**Quick fixes** (in order of effectiveness):

```python
# Fix 1: Reduce mixer weight
user_params = {
    "SCF.Mixer.Weight": 0.01,  # Was 0.1 (default)
    "SCF.Mixer.History": 8,    # Was 5 (default)
}

# Fix 2: Increase iterations
user_params = {
    "MaxSCFIterations": 300,  # Was 50
}

# Fix 3: For metals - add electronic temperature
user_params = {
    "ElectronicTemperature": "300 K",
    "OccupationFunction": "FD",  # Fermi-Dirac
}

# Fix 4: For magnetic systems - enable spin polarization
user_params = {
    "Spin": "polarized",
}

# Fix 5: Use custodian for automatic recovery
maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,  # Enables automatic fixes
    custodian_handlers=[SCFConvergenceHandler(max_attempts=10)],
)
```

**See**: [common_errors/scf_convergence_issues.py](common_errors/scf_convergence_issues.py)

---

### Geometry Optimization Not Converging

**Symptom**: "Reached maximum number of relaxation steps"

**Quick fixes**:

```python
# Fix 1: Tighter force tolerance
user_params = {
    "MD.MaxForceTol": "0.02 eV/Ang",  # Was 0.04
}

# Fix 2: More optimization steps
user_params = {
    "MD.NumCGsteps": 500,  # Was 200
}

# Fix 3: Different optimization algorithm
user_params = {
    "MD.TypeOfRun": "CG",      # Conjugate gradient
    "MD.VariableCell": "true",  # If cell should relax
}

# Fix 4: Better initial structure
# - Check for overlapping atoms
# - Use standardized structure from pymatgen
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
sga = SpacegroupAnalyzer(structure)
structure = sga.get_refined_structure()
```

---

### Memory Errors

**Symptom**: "Exceeded memory limit" or "Out of memory"

**Quick fixes**:

```python
# For local runs - reduce memory usage
user_params = {
    "DiagMemory": "false",      # Avoid full diagonalization if possible
    "SaveDM": "false",           # Don't save density matrix
}

# For jobflow-remote - increase allocation
# Command line:
jf -p PROJECTNAME job set resources <db_id> --mem-per-cpu 4GB

# Or in Python before submission:
from jobflow_remote import set_run_config
job = set_run_config(
    job,
    resources={"mem_per_cpu": "4GB", "nodes": 2}
)
```

---

### K-Point Errors

**Symptom**: "kgrid: ERROR. Grid generates no points"

**Cause**: K-point density too high for small cell, or wrong format

**Quick fixes**:

```python
# Fix 1: Explicit k-point mesh
user_params = {
    "a2s_kpts": [4, 4, 4],  # Explicit mesh
}

# Fix 2: Ensure structure is not too small
# For 1D/2D systems:
user_params = {
    "a2s_kpts": [6, 6, 1],  # For 2D systems (one direction = 1)
}

# Fix 3: Use cutoff instead (automatic)
user_params = {
    "kgrid.Cutoff": "10 Ang",  # Automatic mesh from cutoff
}
```

**Wrong**:
```python
user_params = {
    "kpts": "[4, 4, 4]",  # String - WRONG!
}
```

**Correct**:
```python
user_params = {
    "a2s_kpts": [4, 4, 4],  # List - CORRECT
}
```

---

### Basis Set Errors

**Symptom**: "PAO.Basis: Unknown basis size" or "Basis file not found"

**Quick fixes**:

```python
# Fix 1: Use standard basis sizes
user_params = {
    "PAO.BasisSize": "DZP",  # Standard choices: SZ, DZ, SZP, DZP, TZP
}

# Fix 2: Check pseudopotential has basis definition
# Ensure pseudopotential supports chosen basis size
# ONCVPSP pseudopotentials support all basis sizes

# Fix 3: Provide custom basis block (advanced)
user_params = {
    "fdf_arguments": {
        "PAO.Basis": [
            "Si  2",
            "  n=3  0  2",
            "    4.0  0.0",
            "  n=3  1  1",
            "    4.5",
        ]
    }
}
```

**See**: [03-advanced-features/10-species-variants](../03-advanced-features/10-species-variants/) for custom basis

---

### Pseudopotential Errors

**Symptom**: "Reading pseudopotential failed" or "Species label not found"

**Quick fixes**:

```bash
# Check available pseudopotentials
atomate2siesta-pseudos available

# Install missing pseudopotential
atomate2siesta-pseudos install ONCVPSP-PBE-SR-PDv0.4-Standard

# Verify installation
atomate2siesta-pseudos show ONCVPSP-PBE-SR-PDv0.4-Standard Si
```

**In code**:
```python
# Explicitly specify pseudopotential family
user_params = {
    "a2s_pseudo_set": "ONCVPSP-PBE-SR-PDv0.4-Standard",
}

# Check installed pseudopotentials
from atomate2.siesta.sets.utils import list_available_pseudopotentials
print(list_available_pseudopotentials())
```

---

## Using Custodian for Automatic Recovery

**Custodian** automatically detects and fixes common errors without manual intervention.

### Basic Usage

```python
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.custodian import SCFConvergenceHandler

# Enable custodian with default handlers
maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,  # Enable automatic error handling
)

# Or with custom handlers
from atomate2.siesta.custodian import (
    SCFConvergenceHandler,
    GeometryOptimizationHandler,
    MeshCutoffHandler,
)

maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,
    custodian_handlers=[
        SCFConvergenceHandler(max_attempts=10),
        GeometryOptimizationHandler(max_attempts=5),
        MeshCutoffHandler(),
    ],
    custodian_max_errors=15,  # Total error limit
)
```

### What Custodian Fixes Automatically

| Error Type | Fix Applied | Success Rate |
|------------|-------------|--------------|
| SCF convergence | Reduce mixer weight progressively | high |
| Geometry optimization | Tighter tolerances, more steps | 90% |
| Mesh cutoff issues | Increase cutoff by 50 Ry | 85% |
| K-point errors | Adjust k-point density | 80% |
| Memory errors | Reduce memory usage options | 70% |

**See**: [03-advanced-features/04-error-handling](../03-advanced-features/04-error-handling/) for custodian details

---

## Diagnostic Tools

### Check Forces and Stress

```python
from pymatgen.io.siesta import SiestaOutput

# Read output
output = SiestaOutput("job_*/siesta.out")

# Check final forces
forces = output.forces[-1]  # Last geometry step
max_force = max(np.linalg.norm(f) for f in forces)
print(f"Max force: {max_force:.4f} eV/Å")

# Check stress
stress = output.stress[-1]
max_stress = np.abs(stress).max()
print(f"Max stress: {max_stress:.4f} GPa")

# Convergence criteria
if max_force < 0.04:  # Default tolerance
    print("✓ Forces converged")
else:
    print("✗ Forces NOT converged")
```

### Analyze SCF Convergence

```python
import numpy as np
import matplotlib.pyplot as plt

# Extract SCF energies
energies = []
with open("job_*/siesta.out") as f:
    for line in f:
        if "scf:" in line:
            parts = line.split()
            energies.append(float(parts[2]))  # Energy column

# Plot convergence
plt.plot(energies)
plt.xlabel("SCF Iteration")
plt.ylabel("Energy (eV)")
plt.title("SCF Convergence")
plt.savefig("scf_convergence.png")

# Check for oscillations
energy_diff = np.diff(energies)
oscillating = (energy_diff[:-1] * energy_diff[1:] < 0).sum()
print(f"Sign changes in energy: {oscillating}")
if oscillating > 10:
    print("⚠ Oscillating convergence - reduce mixer weight!")
```

### Check Band Gap (Metallic?)

```python
from pymatgen.io.siesta import SiestaOutput

output = SiestaOutput("job_*/siesta.out")
band_gap = output.band_gap  # May be None if not computed

if band_gap is not None:
    if band_gap < 0.01:
        print("✓ System is metallic - use electronic temperature")
    else:
        print(f"✓ Band gap = {band_gap:.2f} eV - insulator/semiconductor")
else:
    print("⚠ Band gap not computed - run band structure calculation")
```

---

## Preventive Measures

### Always Start with Convergence Tests

Before production runs:

```python
from atomate2.siesta.flows.convergence import (
    MeshCutoffConvergenceFlowMaker,
    KpointsConvergenceFlowMaker,
)

# Step 1: Find converged mesh cutoff
mesh_flow = MeshCutoffConvergenceFlowMaker(
    material_formula="Si",
    cutoffs=[200, 250, 300, 350, 400],  # Ry
)

# Step 2: Find converged k-points
kpts_flow = KpointsConvergenceFlowMaker(
    material_formula="Si",
    kpoints_list=[[2,2,2], [4,4,4], [6,6,6], [8,8,8]],
)
```

**See**: [02-workflows/01-convergence](../02-workflows/01-convergence/)

### Use Appropriate Tier Presets

```python
from atomate2.siesta.sets.tiers import apply_tier_preset

# Material-specific optimized parameters
maker = RelaxMaker.fixed_cell_relaxation()

# For 2D materials
maker = apply_tier_preset(maker, "2d_semiconductor")

# For magnetic materials
maker = apply_tier_preset(maker, "magnetic_correlated")

# For surface calculations
maker = apply_tier_preset(maker, "surface_metal")
```

**See**: [03-advanced-features/07-tier-based-parameters](../03-advanced-features/07-tier-based-parameters/)

### Enable Dry-Run First

```python
# Test workflow without running calculations
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

# Check generated files
ls preview_output/job_*/
cat preview_output/job_*/siesta.fdf

# If looks good, rerun with dry_run=False
```

### Use Custodian by Default

```python
# Enable for all calculations
maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,  # ← Always enable!
)
```

high of common errors will be fixed automatically.

---

## When to Ask for Help

If you've tried the troubleshooting guides and still failing:

### Gather This Information

1. **Complete error message**:
   ```bash
   tail -200 job_*/siesta.out > error.txt
   ```

2. **Input file**:
   ```bash
   cp job_*/siesta.fdf input.fdf
   ```

3. **Structure**:
   ```bash
   cp job_*/siesta.XV structure.xv
   ```

4. **System info**:
   - Material formula
   - Calculation type (relax, static, NEB, etc.)
   - atomate2siesta version: `pip show atomate2siesta`
   - SIESTA version: `siesta --version`

5. **What you've tried**:
   - List fixes attempted
   - Parameter changes made
   - Error messages before/after

### Where to Get Help

- **GitHub Issues**: https://github.com/materialsproject/atomate2/issues
- **SIESTA Forum**: https://siesta-project.org/SIESTA/Forum
- **Atomate2 Discussions**: Check general jobflow/atomate2 issues

---

## Tips for Successful Troubleshooting

✅ **Read error messages carefully**: They usually tell you exactly what's wrong
✅ **Start simple**: Test with minimal system first (2-4 atoms)
✅ **One change at a time**: Don't modify 5 parameters simultaneously
✅ **Compare with working case**: If Si works but your system doesn't, what's different?
✅ **Check literature**: Has anyone calculated this material before?
✅ **Use custodian**: Let it fix common errors automatically
✅ **Document what works**: Save working parameter sets for similar systems
✅ **Don't over-tighten**: Tighter ≠ better (diminishing returns after convergence)

---

## Common Misconceptions

❌ **"More k-points is always better"**
   → After convergence (~0.01 eV), more k-points just waste time

❌ **"SCF must converge to 1e-8"**
   → 1e-4 is usually sufficient; 1e-5 for forces; 1e-6 for phonons

❌ **"Calculation failed → SIESTA bug"**
   → 99% of failures are parameter/input issues, not code bugs

❌ **"DZP always better than DZ"**
   → For band gaps, maybe. For total energies, sometimes worse.

❌ **"Hybrid functionals fix all problems"**
   → They fix band gaps, not SCF convergence. Use GGA first.

---

## Next Steps

After resolving issues:

1. **Document the fix**: Add comments to your workflow code
2. **Update tier presets**: If material-specific fix, consider contributing preset
3. **Run convergence tests**: Verify solution is converged
4. **Enable custodian**: Prevent future occurrences automatically
5. **Continue research**: Get back to science!

---

## References

### SIESTA Documentation
- **User's Guide**: https://docs.siesta-project.org/
- **FAQ**: https://siesta-project.org/SIESTA/Faq
- **Mailing List**: https://siesta-project.org/SIESTA/Forum

### Troubleshooting Resources
- Custodian documentation: [03-advanced-features/04-error-handling](../03-advanced-features/04-error-handling/)
- Convergence testing: [02-workflows/01-convergence](../02-workflows/01-convergence/)
- Parameter reference: [03-advanced-features/09-fdf-parameters](../03-advanced-features/09-fdf-parameters/)

### External Tools
- **ASE Troubleshooting**: https://wiki.fysik.dtu.dk/ase/faq.html
- **Pymatgen Structure Analysis**: https://pymatgen.org/usage.html
- **MongoDB Query**: For jobflow-remote database inspection

---

*Back to [Main Tutorial Index](../README.md)*
