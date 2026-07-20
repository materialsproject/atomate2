# Powerups Tutorial

## Overview

**Powerups** are functions that modify jobs and workflows **after creation** to customize parameters, add functionality, or build high-throughput screening workflows. They provide flexible post-creation modifications without changing maker definitions.

**Why use powerups?**
- ✅ **Flexible**: Modify jobs after creation
- ✅ **Reusable**: Apply same changes to multiple jobs
- ✅ **Clean**: Separate workflow logic from parameters
- ✅ **Powerful**: Enable high-throughput screening
- ✅ **Chainable**: Combine multiple powerups

---

## Core Powerups

### 1. `update_user_siesta_settings()`

Update SIESTA parameters for jobs or workflows.

```python
from atomate2.siesta.powerups import update_user_siesta_settings

# Update single job
job = RelaxMaker.fixed_cell_relaxation().make(structure)
job = update_user_siesta_settings(
    job,
    {
        "PAO.BasisSize": "DZP",
        "kpts": [6, 6, 6],
        "Mesh.Cutoff": "300 Ry",
    }
)

# Update entire workflow
flow = KpointsConvergenceMaker(...).make(structure)
flow = update_user_siesta_settings(
    flow,
    {"PAO.BasisSize": "DZP"}  # Applied to ALL jobs in flow
)
```

### 2. `add_metadata()`

Attach custom metadata to jobs for tracking and organization.

```python
from atomate2.siesta.powerups import add_metadata

job = add_metadata(
    job,
    {
        "project": "surface_energy",
        "material": "Si",
        "version": "v2.0",
        "notes": "High accuracy calculation"
    }
)
```

### 3. `use_fake_siesta()`

Replace SIESTA with fake calculator for testing workflow logic.

```python
from atomate2.siesta.powerups import use_fake_siesta

# Test workflow without running SIESTA
job = RelaxMaker.fixed_cell_relaxation().make(structure)
job = use_fake_siesta(job)

# Runs instantly, returns fake results
results = run_locally(job)
```

---

## Basic Usage

### Update Parameters After Creation

```python
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.powerups import update_user_siesta_settings

# Create job with defaults
maker = RelaxMaker.fixed_cell_relaxation()
job = maker.make(structure)

# Modify parameters using powerup
job = update_user_siesta_settings(
    job,
    {
        "PAO.BasisSize": "DZP",
        "kpts": [8, 8, 8],
        "Mesh.Cutoff": "350 Ry",
        "SCF.Mixer.Weight": 0.05,
        "SCF.DM.Tolerance": 1.0e-5,
    }
)

# Run with updated parameters
run_locally(job)
```

### Chain Multiple Powerups

```python
# Apply multiple powerups sequentially
job = maker.make(structure)

# 1. Update SIESTA parameters
job = update_user_siesta_settings(job, {"PAO.BasisSize": "DZDP"})

# 2. Add tracking metadata
job = add_metadata(job, {"study": "basis_test", "version": "v1.0"})

# 3. Use fake SIESTA for testing
job = use_fake_siesta(job)

run_locally(job)
```

---

## Flow Customization

### Apply Settings to Entire Workflow

```python
from atomate2.siesta.flows.convergence import KpointsConvergenceMaker

# Create convergence workflow
flow = KpointsConvergenceMaker(
    kpoints_list=[[2,2,2], [4,4,4], [6,6,6], [8,8,8]]
)
workflow = flow.make(structure)

# Apply same basis to ALL jobs in workflow
workflow = update_user_siesta_settings(
    workflow,
    {
        "PAO.BasisSize": "DZP",  # All jobs use DZP
        "Mesh.Cutoff": "300 Ry",  # All jobs use 300 Ry
    }
)

# All 4 k-point tests now use DZP basis + 300 Ry cutoff
run_locally(workflow)
```

### Add Metadata to Workflows

```python
# Track convergence study metadata
workflow = add_metadata(
    workflow,
    {
        "study_type": "kpoints_convergence",
        "material": "Si",
        "target_accuracy": "1 meV",
        "date": "2025-10-22"
    }
)
```

---

## High-Throughput Screening

### Basis Set Screening

```python
# Test multiple basis sets
basis_sets = ["SZ", "DZ", "DZP", "DZDP", "TZP"]

jobs = []
for basis in basis_sets:
    # Create job
    job = StaticMaker().make(structure)

    # Apply basis-specific parameters
    job = update_user_siesta_settings(
        job,
        {"PAO.BasisSize": basis}
    )

    # Add metadata
    job = add_metadata(job, {"basis_test": basis})

    jobs.append(job)

# Run all basis tests
for job in jobs:
    run_locally(job, create_folders=True)
```

### Multi-Structure Screening

```python
# Screen multiple materials
structures = {
    "Si": Structure.from_file("Si.cif"),
    "Ge": Structure.from_file("Ge.cif"),
    "GaAs": Structure.from_file("GaAs.cif"),
}

jobs = []
for formula, struct in structures.items():
    # Create job
    job = RelaxMaker.fixed_cell_relaxation().make(struct)

    # Apply common parameters
    job = update_user_siesta_settings(
        job,
        {
            "PAO.BasisSize": "DZP",
            "kpts": [6, 6, 6],
            "Mesh.Cutoff": "300 Ry",
        }
    )

    # Track material
    job = add_metadata(job, {"formula": formula, "screening": True})

    jobs.append(job)

# Submit all jobs
from jobflow_remote import submit_flow

for job in jobs:
    job_id = submit_flow(job, project="screening", worker="slurm_worker")
    print(f"Submitted {job.metadata['formula']}: {job_id}")
```

### Parameter Grid Search

```python
# Test combinations of parameters
energy_shifts = [0.01, 0.02, 0.03]  # Ry
split_norms = [0.10, 0.15, 0.20]

jobs = []
for shift in energy_shifts:
    for norm in split_norms:
        job = RelaxMaker.fixed_cell_relaxation().make(structure)

        job = update_user_siesta_settings(
            job,
            {
                "PAO.EnergyShift": f"{shift} Ry",
                "PAO.SplitNorm": norm,
            }
        )

        job = add_metadata(
            job,
            {
                "energy_shift": shift,
                "split_norm": norm,
                "grid_search": True
            }
        )

        jobs.append(job)

print(f"Total combinations: {len(jobs)}")  # 3 × 3 = 9 jobs
```

---

## Advanced Patterns

### Conditional Parameter Updates

```python
def customize_by_element(job, structure):
    """Apply element-specific parameters."""
    formula = structure.composition.reduced_formula

    if "O" in formula:
        # Oxygen-containing: tighter parameters
        job = update_user_siesta_settings(
            job,
            {
                "PAO.BasisSize": "DZDP",
                "SCF.DM.Tolerance": 1.0e-6,
            }
        )
    elif formula in ["Si", "Ge"]:
        # Semiconductors: standard parameters
        job = update_user_siesta_settings(
            job,
            {
                "PAO.BasisSize": "DZP",
                "SCF.DM.Tolerance": 1.0e-5,
            }
        )
    else:
        # Others: relaxed parameters
        job = update_user_siesta_settings(
            job,
            {
                "PAO.BasisSize": "DZ",
                "SCF.DM.Tolerance": 1.0e-4,
            }
        )

    return job

# Apply conditional updates
job = maker.make(structure)
job = customize_by_element(job, structure)
```

### Workflow Templates

```python
def apply_production_settings(workflow):
    """Standard production settings for all workflows."""
    workflow = update_user_siesta_settings(
        workflow,
        {
            "PAO.BasisSize": "DZP",
            "Mesh.Cutoff": "300 Ry",
            "SCF.DM.Tolerance": 1.0e-5,
        }
    )

    workflow = add_metadata(
        workflow,
        {
            "quality": "production",
            "validated": True,
            "version": "2.0"
        }
    )

    return workflow

# Apply to any workflow
flow1 = KpointsConvergenceMaker(...).make(structure)
flow1 = apply_production_settings(flow1)

flow2 = EOSMaker(...).make(structure)
flow2 = apply_production_settings(flow2)
```

---

## Best Practices

### 1. Use Powerups for Customization

```python
# ✅ GOOD: Use powerups for customization
job = maker.make(structure)
job = update_user_siesta_settings(job, {"PAO.BasisSize": "DZP"})

# ❌ AVOID: Modifying maker directly (less flexible)
maker = RelaxMaker.fixed_cell_relaxation(
    user_params={"PAO.BasisSize": "DZP"}
)
```

### 2. Separate Logic from Parameters

```python
# ✅ GOOD: Create workflow logic, then apply parameters
flow = KpointsConvergenceMaker(kpoints_list=[...])
workflow = flow.make(structure)
workflow = update_user_siesta_settings(workflow, common_params)

# ❌ AVOID: Mixing workflow logic with parameter details
```

### 3. Add Metadata for Tracking

```python
# ✅ GOOD: Track study details
job = add_metadata(
    job,
    {
        "project": "surfaces",
        "material": formula,
        "date": "2025-10-22",
        "parameters": {"basis": "DZP", "cutoff": "300 Ry"}
    }
)
```

### 4. Test with Fake SIESTA First

```python
# ✅ GOOD: Validate workflow logic before running
job = use_fake_siesta(job)
results = run_locally(job)  # Instant
# ... verify workflow structure ...

# Then run real calculation
job = maker.make(structure)  # Without fake SIESTA
results = run_locally(job)
```

---

## Integration with Other Features

### With Dry-Run Mode

```python
# Preview parameters after powerup
job = maker.make(structure)
job = update_user_siesta_settings(job, {"PAO.BasisSize": "DZDP"})

# Dry-run to inspect
job_preview = RelaxMaker.fixed_cell_relaxation(dry_run=True).make(structure)
job_preview = update_user_siesta_settings(job_preview, {"PAO.BasisSize": "DZDP"})
run_locally(job_preview)

# Check preview_output/*/siesta.fdf for DZDP basis
```

### With Tier System

```python
from atomate2.siesta.sets.tiers import apply_tier_preset

# Combine tier presets with powerups
job = maker.make(structure)
job = apply_tier_preset(job, "relax_standard")  # Apply tier first
job = update_user_siesta_settings(job, {"kpts": [10, 10, 10]})  # Override k-points
```

### With Database Storage

```python
# Metadata persists in MongoDB
job = add_metadata(
    job,
    {
        "study": "screening",
        "batch": "batch_01",
        "priority": "high"
    }
)

# Query later by metadata
from maggma.stores import MongoStore

store = MongoStore(...)
docs = list(store.query({"metadata.study": "screening"}))
```

---

## Summary

### Key Takeaways

✅ **Powerups modify jobs AFTER creation** (post-processing)
✅ **update_user_siesta_settings()** updates any SIESTA parameter
✅ **Works on jobs or workflows** (applies to all child jobs)
✅ **add_metadata()** attaches custom tracking information
✅ **Chainable** - combine multiple powerups
✅ **Perfect for screening** - loop + powerup pattern

### Essential Powerups

```python
# Update parameters
update_user_siesta_settings(job, params_dict)

# Add metadata
add_metadata(job, metadata_dict)

# Test workflow logic
use_fake_siesta(job)
```

### Quick Reference

```python
# Single job
job = maker.make(structure)
job = update_user_siesta_settings(job, {"PAO.BasisSize": "DZP"})
job = add_metadata(job, {"test": "screening"})

# Entire workflow
flow = KpointsConvergenceMaker(...).make(structure)
flow = update_user_siesta_settings(flow, {"Mesh.Cutoff": "300 Ry"})

# High-throughput screening
for param in parameter_list:
    job = maker.make(structure)
    job = update_user_siesta_settings(job, param)
    jobs.append(job)
```

### Next Steps

1. **Try the tutorial**: `python tutorial.py`
2. **Test example types**:
   - `'basic'`: Parameter updates
   - `'flow'`: Workflow customization
   - `'screening'`: High-throughput screening
3. **Build your own screening workflows**
4. **Combine with database storage** for data management

---

**Make powerups part of your workflow customization strategy!** ⚡
