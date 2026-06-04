# Restart and Recovery Tutorial

## Overview

Atomate2-siesta provides automatic restart capabilities through the `prev_dir` parameter and custodian error handling.

## Tutorial Scripts

| Script | Description | Key Concepts |
|--------|-------------|--------------|
| `01_basic_restart.py` | Basic restart from previous calculation | prev_dir, density matrix reuse |
| `02_interrupted_calculation.py` | Recovery from walltime/crash | Finding directories, continuing progress |
| `03_manual_recovery.py` | Manual parameter adjustment | Diagnosis, parameter fixes, powerups |
| `04_comparison_custodian.py` | Manual vs automatic recovery | Why custodian is recommended |

**Run the tutorials:**

```bash
cd tutorials/04-infrastructure/04-restart-recovery
python3 01_basic_restart.py           # Basic restart example
python3 02_interrupted_calculation.py # Interrupted job recovery
python3 03_manual_recovery.py         # Manual parameter fixes
python3 04_comparison_custodian.py    # Custodian vs manual
```

## Restart from Previous Calculation

### Basic Restart

```python
from atomate2.siesta.jobs.core import RelaxMaker

# Initial calculation
maker = RelaxMaker.fixed_cell_relaxation()
job1 = maker.make(structure)
results = run_locally(job1, create_folders=True)

# Restart from previous directory
prev_dir = "path/to/job_*/outputs"
job2 = maker.make(structure, prev_dir=prev_dir)
results2 = run_locally(job2)
```

### What Gets Reused

- ✅ Density matrix (.DM file)
- ✅ Converged electronic structure
- ✅ Optimized geometry (if relaxation)
- ✅ Reduces SCF iterations
- ✅ Faster convergence

## Automatic Error Recovery

Use custodian for automatic error handling:

```python
from atomate2.siesta.jobs.core import RelaxMaker

# Enable custodian (automatic retry on failures)
maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,
    custodian_max_errors=10  # Max retry attempts
)

job = maker.make(structure)
results = run_locally(job)

# Custodian automatically:
# - Detects SCF convergence failures
# - Adjusts mixer parameters
# - Retries calculation
# - Logs all corrections in custodian.json
```

## Error Handling Features

See **custodian tutorial** for comprehensive error handling:

**Tutorial**: `04-infrastructure/03-error-handling/`

Covers:
- ✅ 4 default handlers (SCF, Walltime, Memory, Frozen)
- ✅ Progressive 5-level SCF correction strategy
- ✅ Automatic parameter adjustment
- ✅ Custom handler creation
- ✅ custodian.json logging

## Manual Recovery Steps

### 1. Identify Failure

```bash
# Check output file
tail siesta.out

# Common issues:
# - "SCF not converged"
# - "WALLTIME exceeded"
# - "Memory allocation failed"
```

### 2. Adjust Parameters

```python
from atomate2.siesta.powerups import update_user_siesta_settings

# Create recovery job
maker = RelaxMaker.fixed_cell_relaxation()
job = maker.make(structure, prev_dir="failed_job/outputs")

# Apply fixes
job = update_user_siesta_settings(
    job,
    {
        "SCF.Mixer.Weight": 0.01,  # Slower mixing
        "MaxSCFIterations": 200,  # More iterations
        "SCF.DM.Tolerance": 1.0e-4,  # Relaxed tolerance
    }
)

results = run_locally(job)
```

### 3. Use Custodian

Better approach - let custodian handle it automatically:

```python
# Just enable custodian - handles most issues automatically
maker = RelaxMaker.fixed_cell_relaxation(use_custodian=True)
job = maker.make(structure)
run_locally(job)  # Automatic recovery!
```

## Best Practices

1. **Always enable custodian** for production calculations
2. **Check custodian.json** to see what corrections were applied
3. **Use prev_dir** for manual restarts to reuse converged data
4. **Start with dry-run** to validate parameters before expensive calculations

## See Also

- Custodian tutorial: `04-infrastructure/03-error-handling/`
- Powerups tutorial: `07-advanced-features/03-powerups/`

**Use custodian for automatic error recovery in all production workflows!**
