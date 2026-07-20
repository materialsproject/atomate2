# Tutorial 14: Efficiency and Performance Monitoring

## Overview

This tutorial demonstrates how to configure SIESTA's efficiency and performance monitoring options using the `EfficiencyOptions` dataclass module. These settings help optimize calculations, monitor resource usage, and manage long-running jobs on HPC clusters.

## Learning Objectives

- Configure memory allocation reporting for debugging
- Enable timing analysis for performance profiling
- Set walltime limits for cluster job management
- Optimize calculation efficiency with DirectPhi
- Use restart data for interrupted calculations

## Theory

### Performance Monitoring in SIESTA

SIESTA provides comprehensive monitoring tools to help identify bottlenecks and optimize calculations:

1. **Memory Allocation Reporting**: Track memory usage at different verbosity levels
2. **Timing Analysis**: Measure time spent in different code sections
3. **Walltime Management**: Gracefully handle cluster time limits
4. **Calculation Optimization**: Skip unnecessary intermediate steps

### Why Monitor Performance?

**Memory Issues:**
- Identify memory leaks and allocation patterns
- Optimize memory usage for large systems
- Debug out-of-memory errors

**Performance Bottlenecks:**
- Find slow code sections (diagonalization, grid operations, etc.)
- Balance load across MPI processes
- Optimize parallelization strategy

**Job Management:**
- Ensure calculations complete before cluster time limits
- Create restart files automatically
- Avoid wasted compute time

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `direct_phi` | bool | False | Skip intermediate phi calculations |
| `alloc_report_level` | int | 0 | Memory allocation verbosity (0-3) |
| `alloc_report_threshold` | float | 0.0 | Min memory (MB) to report |
| `timer_report_threshold` | float | 0.0 | Min time (s) to report |
| `user_tree_timer` | bool | False | Enable hierarchical timing tree |
| `user_parallel_timer` | bool | False | Per-process timing reports |
| `timing_split_scf_steps` | bool | False | Time individual SCF steps |
| `max_walltime` | float | None | Maximum runtime (seconds) |
| `max_walltime_slack` | float | 5.0 | Safety margin (seconds) |
| `use_save_data` | bool | False | Use restart data if available |

## Examples

### Example 1: Memory Allocation Reporting

Monitor memory usage during calculation:

```python
user_params = {
    "AllocReportLevel": 2,          # Moderate verbosity
    "AllocReportThreshold": 10.0,   # Report allocations >10 MB
}

maker = RelaxMaker.fixed_cell_relaxation(user_params=user_params)
```

**AllocReportLevel options:**
- **0**: No reporting (default)
- **1**: Major allocations only
- **2**: Moderate verbosity (recommended)
- **3**: All allocations (very verbose)

**When to use:**
- Debugging memory issues
- Optimizing large calculations
- Identifying memory-intensive operations

### Example 2: Walltime Limits for Cluster Jobs

Set maximum runtime with automatic restart:

```python
user_params = {
    "MaxWalltime": 3500.0,          # 58 min 20 sec
    "MaxWalltimeSlack": 100.0,      # Stop 100s early
    "UseSaveData": True,            # Enable restart
}

maker = RelaxMaker.fixed_cell_relaxation(user_params=user_params)
```

**Best practices:**
- Set MaxWalltime to 90-high of cluster limit
- Use slack time for clean shutdown
- Enable UseSaveData for continuation

**Example: 1-hour job queue**
```python
cluster_limit = 3600  # seconds
max_walltime = 3600 - 200  # 3400s (leave 200s margin)
max_walltime_slack = 100   # Stop 100s early for cleanup
# Total runtime: 3300s (55 minutes)
```

### Example 3: Comprehensive Timing Analysis

Enable detailed performance profiling:

```python
user_params = {
    "UserTreeTimer": True,          # Hierarchical timing tree
    "UserParallelTimer": True,      # Per-process timing
    "TimingSplitSCFSteps": True,    # Time each SCF iteration
    "TimerReportThreshold": 0.01,   # Report times >10ms
}

maker = StaticMaker.scf(user_params=user_params)
```

**Timing output includes:**
- Total time per major section (setup, SCF, forces)
- Per-process timing for load balance analysis
- Individual SCF step timing
- Grid operations, diagonalization, communication

## Memory Allocation Reporting

### Configuration Levels

**Level 0 (None):**
```python
{"AllocReportLevel": 0}  # Default, no output
```

**Level 1 (Major):**
```python
{
    "AllocReportLevel": 1,
    "AllocReportThreshold": 50.0,  # Only >50 MB
}
```
Output: Major arrays (density matrix, Hamiltonian, grids)

**Level 2 (Moderate):**
```python
{
    "AllocReportLevel": 2,
    "AllocReportThreshold": 10.0,  # Report >10 MB
}
```
Output: Most allocations with reasonable verbosity

**Level 3 (All):**
```python
{
    "AllocReportLevel": 3,
    "AllocReportThreshold": 0.0,  # Report everything
}
```
Output: Every allocation (very verbose, for debugging only)

### Example Output

```
alloc: array rho [      0.039 GB] in node 0
alloc: array phi [      0.025 GB] in node 0
alloc: array H [         0.152 GB] in node 0
alloc: array S [         0.152 GB] in node 0
Total allocated:         0.368 GB
Peak memory usage:       0.521 GB
```

### Best Practices

**Production calculations:**
```python
{
    "AllocReportLevel": 0,  # Disable for performance
}
```

**Development/debugging:**
```python
{
    "AllocReportLevel": 2,
    "AllocReportThreshold": 10.0,
}
```

**Memory issues:**
```python
{
    "AllocReportLevel": 3,
    "AllocReportThreshold": 0.0,
}
```

## Timing Analysis

### Timer Types

**1. Tree Timer (UserTreeTimer)**

Hierarchical timing breakdown:
```
SIESTA execution
├── Setup (2.3 s)
│   ├── Read input (0.5 s)
│   ├── Build basis (1.2 s)
│   └── Initialize grid (0.6 s)
├── SCF (125.4 s)
│   ├── Setup H (15.2 s)
│   ├── Diagonalization (95.3 s)
│   └── Charge mixing (14.9 s)
└── Forces (8.7 s)
```

**2. Parallel Timer (UserParallelTimer)**

Per-process timing for load balancing:
```
Process  | Setup | SCF   | Forces
---------|-------|-------|-------
0        | 2.1s  | 124s  | 8.5s
1        | 2.3s  | 125s  | 8.7s
2        | 2.2s  | 126s  | 8.9s  ← slower process
3        | 2.1s  | 124s  | 8.4s
```

**3. SCF Step Timing (TimingSplitSCFSteps)**

Individual SCF iteration timing:
```
SCF iter | Total | Setup | Diag  | Mix
---------|-------|-------|-------|-----
1        | 12.5s | 3.2s  | 8.1s  | 1.2s
2        | 10.3s | 2.8s  | 6.4s  | 1.1s
3        | 10.1s | 2.7s  | 6.3s  | 1.1s
...
20       | 10.0s | 2.7s  | 6.2s  | 1.1s
```

### Configuration Examples

**Quick profiling:**
```python
{
    "UserTreeTimer": True,
    "TimerReportThreshold": 0.1,  # Skip times <100ms
}
```

**Load balance analysis:**
```python
{
    "UserParallelTimer": True,
    "TimerReportThreshold": 0.01,
}
```

**SCF convergence profiling:**
```python
{
    "TimingSplitSCFSteps": True,
}
```

**Complete profiling:**
```python
{
    "UserTreeTimer": True,
    "UserParallelTimer": True,
    "TimingSplitSCFSteps": True,
    "TimerReportThreshold": 0.01,
}
```

## Walltime Management

### Why Use Walltime Limits?

**Cluster job queues** have time limits:
- Short queue: 1-4 hours
- Medium queue: 12-24 hours
- Long queue: 48-168 hours

**Without walltime limits:**
- Job killed abruptly when time expires
- No restart files saved
- Wasted compute time

**With walltime limits:**
- Calculation stops gracefully
- Restart files saved automatically
- Can continue in next job

### Configuration Strategy

**1. Determine cluster time limit:**
```bash
# Example: 1-hour queue
cluster_limit = 3600 seconds
```

**2. Set MaxWalltime with safety margin:**
```python
# Leave 5-10% margin for shutdown
max_walltime = cluster_limit * 0.90  # 3240s (54 min)
```

**3. Configure slack time:**
```python
# Additional time for cleanup
max_walltime_slack = 100  # seconds
```

**4. Enable restart:**
```python
{
    "MaxWalltime": max_walltime,
    "MaxWalltimeSlack": max_walltime_slack,
    "UseSaveData": True,
}
```

### Example Configurations

**1-hour queue:**
```python
{
    "MaxWalltime": 3200.0,      # 53 min 20 sec
    "MaxWalltimeSlack": 100.0,  # Stop at 51m40s
    "UseSaveData": True,
}
```

**12-hour queue:**
```python
{
    "MaxWalltime": 41400.0,     # 11.5 hours
    "MaxWalltimeSlack": 300.0,  # 5-minute margin
    "UseSaveData": True,
}
```

**Development (no limit):**
```python
{
    "MaxWalltime": None,        # Disabled
}
```

### Restart Workflow

**First job (incomplete):**
```python
maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "MaxWalltime": 3200.0,
        "UseSaveData": True,
    }
)
job = maker.make(structure)
results = run_locally(job)  # Times out, saves restart
```

**Second job (continuation):**
```python
# Uses prev_dir to load restart files
job2 = maker.make(structure, prev_dir="./path/to/first/job")
results = run_locally(job2)  # Continues from saved state
```

## Calculation Optimization

### DirectPhi Option

**What it does:**
- Skips intermediate electrostatic potential (φ) calculations
- Only computes final φ after SCF convergence

**When to use:**
```python
{
    "DirectPhi": True,  # Faster SCF, no intermediate φ
}
```

**Benefits:**
- ✅ Faster SCF iterations (10-20% speedup)
- ✅ Reduced I/O operations
- ✅ Less memory usage

**Drawbacks:**
- ❌ No intermediate φ analysis
- ❌ Cannot monitor potential convergence

**Recommendation:**
- Enable for production calculations
- Disable for debugging or analysis

### UseSaveData Option

**What it does:**
- Reads density matrix from previous calculation
- Accelerates SCF convergence (better initial guess)

**When to use:**
```python
{
    "UseSaveData": True,
}
```

**Use cases:**
1. **Continuation runs**: Restart after walltime limit
2. **Similar structures**: Use converged density as starting point
3. **Multi-step workflows**: EOS, phonons, elastic constants

**Example: EOS workflow**
```python
# First calculation: from scratch
job1 = maker.make(structure_V0)

# Subsequent calculations: use previous density
for i, structure in enumerate(structures[1:]):
    job = maker.make(structure, prev_dir=f"./job{i}")
```

## Performance Benchmarks

### Memory Reporting Overhead

| Level | Overhead | Output Size |
|-------|----------|-------------|
| 0     | 0%       | None        |
| 1     | <1%      | ~10 KB      |
| 2     | ~2%      | ~100 KB     |
| 3     | ~5%      | ~1 MB       |

**Recommendation**: Level 2 for development, level 0 for production

### Timing Analysis Overhead

| Timer Type | Overhead | Use Case |
|------------|----------|----------|
| None       | 0%       | Production |
| Tree       | <1%      | Bottleneck identification |
| Parallel   | ~2%      | Load balance analysis |
| SCF Split  | <1%      | SCF convergence study |
| All        | ~3%      | Complete profiling |

**Recommendation**: Enable only when needed, disable for production

### DirectPhi Performance

**100-atom system, DZP basis:**

| DirectPhi | SCF Time | I/O Operations |
|-----------|----------|----------------|
| False     | 120 s    | 250           |
| True      | 100 s    | 150           |
| **Speedup** | **17%** | **40% fewer** |

## Common Issues

### Issue 1: Memory Allocation Reports Too Verbose

**Problem:** Level 3 produces GB of output

**Solution:**
```python
# Use level 2 with higher threshold
{
    "AllocReportLevel": 2,
    "AllocReportThreshold": 50.0,  # Only large allocations
}
```

### Issue 2: Job Killed Before Walltime

**Problem:** MaxWalltime too close to cluster limit

**Solution:**
```python
# Increase safety margins
{
    "MaxWalltime": cluster_limit * 0.85,  # 15% margin
    "MaxWalltimeSlack": 300.0,            # 5-minute buffer
}
```

### Issue 3: Restart Not Working

**Problem:** UseSaveData=True but calculation starts from scratch

**Solutions:**
1. Check restart files exist (*.DM, *.HSX)
2. Verify prev_dir path is correct
3. Ensure system is identical (atoms, cell, parameters)

### Issue 4: Timing Overhead Too High

**Problem:** Timers slow down calculation

**Solution:**
```python
# Use higher threshold to skip short operations
{
    "UserTreeTimer": True,
    "TimerReportThreshold": 0.5,  # Only report times >0.5s
}
```

## Best Practices

### 1. Development vs Production

**Development:**
```python
{
    "AllocReportLevel": 2,
    "AllocReportThreshold": 10.0,
    "UserTreeTimer": True,
    "TimerReportThreshold": 0.1,
}
```

**Production:**
```python
{
    "AllocReportLevel": 0,
    "DirectPhi": True,
    "UseSaveData": True,  # If applicable
}
```

### 2. Cluster Job Configuration

**Always set for cluster jobs:**
```python
{
    "MaxWalltime": cluster_limit * 0.90,
    "MaxWalltimeSlack": 120.0,
    "UseSaveData": True,
}
```

### 3. Performance Profiling

**Step 1: Identify bottleneck**
```python
{"UserTreeTimer": True}
```

**Step 2: Analyze load balance**
```python
{"UserParallelTimer": True}
```

**Step 3: Study SCF convergence**
```python
{"TimingSplitSCFSteps": True}
```

### 4. Memory Optimization

**Step 1: Monitor usage**
```python
{"AllocReportLevel": 2, "AllocReportThreshold": 10.0}
```

**Step 2: Identify large arrays**
- Grid size (reduce Mesh.Cutoff if needed)
- Basis size (consider smaller basis)
- K-point sampling (reduce if possible)

**Step 3: Optimize parallelization**
- More MPI processes = less memory per process
- Consider hybrid MPI+OpenMP

## Advanced Topics

### Combining Multiple Options

**Complete monitoring setup:**
```python
user_params = {
    # Memory monitoring
    "AllocReportLevel": 2,
    "AllocReportThreshold": 10.0,

    # Timing analysis
    "UserTreeTimer": True,
    "UserParallelTimer": True,
    "TimingSplitSCFSteps": True,
    "TimerReportThreshold": 0.05,

    # Job management
    "MaxWalltime": 3200.0,
    "MaxWalltimeSlack": 100.0,
    "UseSaveData": True,

    # Optimization
    "DirectPhi": True,
}
```

### Parsing Timing Output

**Extract timing data from output:**
```python
import re

def parse_timing_output(output_file):
    """Extract timing information from SIESTA output."""
    with open(output_file) as f:
        lines = f.readlines()

    timings = {}
    for line in lines:
        match = re.search(r'(\w+)\s+:\s+([\d.]+)\s+s', line)
        if match:
            section, time = match.groups()
            timings[section] = float(time)

    return timings

# Example usage
timings = parse_timing_output('siesta.out')
print(f"SCF time: {timings.get('scf', 0):.2f} s")
```

### Memory Usage Estimation

**Estimate memory requirements:**
```python
def estimate_memory(natoms, basis_size="DZP", mesh_cutoff=200):
    """
    Rough memory estimate (GB per process).

    natoms: Number of atoms
    basis_size: Basis set size
    mesh_cutoff: Grid cutoff (Ry)
    """
    # Basis orbitals per atom
    orbitals_per_atom = {"SZ": 13, "DZ": 18, "DZP": 26, "TZP": 39}
    norbs = natoms * orbitals_per_atom.get(basis_size, 26)

    # Matrix memory: H, S, DM (norbs^2 * 8 bytes)
    matrix_mem = 3 * (norbs ** 2) * 8e-9  # GB

    # Grid memory: rho, phi (grid_points * 8 bytes)
    grid_points = (mesh_cutoff / 10) ** 3 * natoms * 200
    grid_mem = 2 * grid_points * 8e-9  # GB

    total = matrix_mem + grid_mem
    return {"matrices": matrix_mem, "grids": grid_mem, "total": total}

# Example: 100 atoms, DZP, 200 Ry
mem = estimate_memory(100, "DZP", 200)
print(f"Estimated memory: {mem['total']:.2f} GB per process")
```

## Further Reading

- SIESTA Manual: Section 6.31 (Memory and Timing)
- HPC Job Management: Best practices for queue systems
- Performance Optimization: SIESTA scaling studies
- Profiling Tools: GNU gprof, Intel VTune

## Next Steps

- Tutorial 15: Hamiltonian and Overlap Matrix Configuration
- Tutorial 12: Parallel Computation Options
- Tutorial 04: Job Submission and Cluster Management
