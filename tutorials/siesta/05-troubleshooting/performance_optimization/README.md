# Performance Optimization

**Category**: troubleshooting/performance_optimization
**Difficulty**: Intermediate
**Time**: 50 minutes total

## Overview

This section provides strategies for optimizing SIESTA calculation performance. Learn how to balance computational cost with accuracy, use parallelization effectively, and reduce overall calculation time.

## Available Tutorials

### 01_parallel_efficiency.py

**Parallel Efficiency**

- K-point parallelization (for dense k-meshes)
- Basis parallelization (for large systems)
- Memory vs speed tradeoffs
- HPC resource allocation guidelines
- Performance benchmarking
- Quick optimization checklist

**Time**: 25 minutes

### 02_computational_cost_reduction.py

**Computational Cost Reduction**

- Understanding cost factors
- Using tier presets for cost control
- When to use tight vs loose convergence
- Calculation hierarchy strategy
- Cost vs accuracy tradeoffs
- Practical cost reduction tips

**Time**: 25 minutes

## Quick Reference

### Cost Factors (in order of impact)

| Factor | Scaling | Impact |
|--------|---------|--------|
| System size | O(N³) | Largest impact |
| K-points | O(N_k) | Linear with k-point count |
| Mesh cutoff | O(N_grid) | Higher cutoff = finer grid |
| Basis size | O(N_orb³) | TZP ~3-5x slower than DZP |

### Recommended Tier Usage

| Application | Tier | Relative Cost |
|-------------|------|---------------|
| Quick screening | basic_dirty | 1x |
| Initial structure | basic | 5x |
| Production | intermediate | 20x |
| High accuracy | advanced | 100x |
| Publication | expert | 500x |

### Parallelization Guidelines

| System Size | Cores | Strategy |
|-------------|-------|----------|
| < 20 atoms | 4-16 | K-point parallel |
| 20-100 atoms | 16-64 | Hybrid |
| 100-500 atoms | 64-256 | Basis parallel |
| > 500 atoms | 256+ | Basis parallel + OrderN |

## Key Optimization Strategies

### 1. Use Tier Presets

```python
from atomate2.siesta.sets.tiers import apply_tier_preset

# Match tier to your needs
maker = apply_tier_preset(maker, "relax_standard")  # Production
maker = apply_tier_preset(maker, "relax_dirty")  # Screening
```

### 2. Follow Calculation Hierarchy

Screening (1 min) → Convergence (30 min) → Production (hours)

### 3. Use Symmetry (default, free speedup)

- Reduces k-points by 2-48x
- No accuracy loss

### 4. Enable Restart Files

```python
user_params = {
    "UseSaveData": True,  # Restart from density matrix
}
```

### 5. Memory Optimization

```python
user_params = {
    "SaveHS": False,
    "WriteWaveFunctions": False,
}
```

## Performance Checklist

Before running:
- Completed convergence tests
- Using appropriate tier preset
- Symmetry enabled (default)
- Tested with dry_run=True

For HPC:
- Cores match k-point count (k-parallel)
- Memory allocation appropriate (4-8 GB/core)
- Wall time estimated correctly

Post-calculation:
- Check timer output for bottlenecks
- Verify parallel efficiency > 70%
- Document optimal parameters

## Common Bottlenecks

| Bottleneck | Symptom | Solution |
|------------|---------|----------|
| SCF iterations | compute_dm dominates | Better initial guess |
| Real-space grid | cellxc dominates | Lower mesh cutoff |
| Diagonalization | Slow with many atoms | Use OrderN |
| I/O | Waiting for file writes | Disable outputs |

## Next Steps

After optimizing performance:

- `../debugging_workflows/` - Troubleshoot issues
- `../../02-workflows/01-convergence/` - Convergence testing
- `../../03-advanced-features/07-tier-based-parameters/` - Tier system details

---

*Back to Troubleshooting Index*
