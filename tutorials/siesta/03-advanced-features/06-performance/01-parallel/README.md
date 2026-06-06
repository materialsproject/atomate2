# Parallel Computation Configuration

## Overview

Configure SIESTA's MPI parallelization for optimal performance on multi-processor systems. These are expert-level parameters that require `tier="expert"`.

## Tutorial Scripts

| Script | Description | Key Parameters |
|--------|-------------|----------------|
| `01_basic_parallel_grid.py` | 2D processor grid configuration | BlockSize, ProcessorY |
| `02_on_decomposition.py` | O(N) domain/spatial decomposition | UseDomainDecomposition, RcSpatial |

**Run the tutorials:**

```bash
cd tutorials/03-advanced-features/06-performance/01-parallel
python 01_basic_parallel_grid.py
python 02_on_decomposition.py
```

---

## Quick Reference

### Basic Processor Grid (Diagonalization)

```python
user_params = {
    "BlockSize": 32,        # 32x32 blocks for matrix distribution
    "ProcessorY": 4,        # 4 processors in Y dimension (16 procs = 4x4 grid)
}

maker = StaticMaker.scf(user_params=user_params, tier="expert")
```

**When to use:**
- < 1000 atoms
- Non-square processor counts
- Default diagonalization solver

### O(N) Domain Decomposition (Large Systems)

```python
user_params = {
    "UseDomainDecomposition": True,
    "UseSpatialDecomposition": True,
    "RcSpatial": 15.0,      # 15 Bohr communication radius
    "SolutionMethod": "OMM", # O(N) solver required
}

maker = RelaxMaker.fixed_cell_relaxation(user_params=user_params, tier="expert")
```

**When to use:**
- > 1000 atoms
- O(N) solvers (OMM, PEXSI)
- Need to scale to many processors

---

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `BlockSize` | int | Auto | Block size for 2D matrix distribution (16-64 typical) |
| `ProcessorY` | int | Auto | Y-dimension of 2D processor grid |
| `UseDomainDecomposition` | bool | False | Group orbitals/atoms by processor (O(N) only) |
| `UseSpatialDecomposition` | bool | True | Distribute real-space grid across processors |
| `RcSpatial` | float | Auto | Communication radius in Bohr (10-30 typical) |

---

## Scaling Guidelines

| System Size | Processors | Method | Settings |
|-------------|-----------|--------|----------|
| < 500 atoms | 4-32 | Diagonalization | Default (automatic) |
| 500-2000 | 32-128 | OMM | Spatial decomposition |
| 2000-10000 | 128-512 | OMM | Domain + spatial |
| > 10000 | 512+ | PEXSI | Domain + spatial + tuned RcSpatial |

---

## Best Practices

### 1. Processor Grid Selection

**Square grids** (ProcessorY = √N):
- ✅ Best for most cases (balanced communication)
- Example: 16 procs = 4×4, 64 procs = 8×8

**Rectangular grids**:
- ✅ Flexibility for any processor count
- Example: 12 procs = 3×4, 24 procs = 4×6

### 2. Block Size Selection

**Small blocks (16-32):**
- Better load balancing
- More communication overhead
- Good for small-medium systems

**Large blocks (64-128):**
- Less communication
- Potential load imbalance
- Good for GPUs

### 3. Start Simple, Then Tune

1. **Start**: Use default settings (no manual config)
2. **Profile**: Check parallel efficiency in output
3. **Tune**: If efficiency < 70%, adjust parameters
4. **Test**: Benchmark different configurations

---

## Common Issues

### Poor Scaling (efficiency < 70%)

**Symptoms:** Time doesn't decrease with more processors

**Solutions:**
- Reduce BlockSize for better load balance
- Adjust ProcessorY for square/rectangular grid
- Enable spatial decomposition

### Memory Overflow

**Symptoms:** Out-of-memory errors

**Solutions:**
- Use more processors (reduces memory/process)
- Enable O(N) decomposition
- Reduce BlockSize

### Communication Bottlenecks

**Symptoms:** Scaling stops at certain processor count

**Solutions:**
- Increase BlockSize to reduce message count
- Use faster interconnect (InfiniBand)
- Check ProcessorY for optimal grid layout

---

## Performance Profiling

Enable detailed timers to check parallel efficiency:

```python
user_params = {
    "UseTreeTimer": True,         # Hierarchical timing
    "TimingSplitScfSteps": True,  # Per-SCF step timing
}
```

Look for in output file:
```
Parallel efficiency: 85.3%   # Good (> 80%)
Load imbalance: 12.1%        # Acceptable (< 15%)
Communication time: 8.5%     # Low overhead (< 10%)
```

---

## Advanced: Hybrid MPI+OpenMP

Combine MPI with OpenMP threading for better memory efficiency:

```bash
export OMP_NUM_THREADS=4
mpirun -n 16 siesta < input.fdf
# 16 MPI × 4 threads = 64 cores total
```

**Benefits:**
- Reduced memory (shared within node)
- Fewer MPI processes = less communication
- Good for memory-limited systems

---

## See Also

- **SIESTA Manual**: Section 6.27 (Parallel Options)
- **Tier System**: `tutorials/03-advanced-features/01-parameter-systems/01-tier-system/`
- **Expert Parameters**: Requires `tier="expert"`
- **ParallelOptions dataclass**: `src/atomate2/siesta/dataclass/parallel_options.py`

---

**Summary:** Start with defaults, profile efficiency, then tune BlockSize/ProcessorY for < 1000 atoms or enable O(N) decomposition for larger systems.
