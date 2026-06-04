"""
Performance Optimization - Parallel Efficiency
==============================================

This tutorial covers parallel efficiency optimization for SIESTA calculations:
- K-point parallelization (best for large systems)
- Basis parallelization (best for small systems)
- Memory vs speed tradeoffs
- HPC resource allocation strategies

Category: troubleshooting/performance_optimization
Difficulty: Intermediate
Time: 25 minutes
"""

from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.core import RelaxMaker

# =============================================================================
# Step 1: Understanding SIESTA Parallelization
# =============================================================================

print("=" * 70)
print("Step 1: Understanding SIESTA Parallelization")
print("=" * 70)

parallelization_overview = """
SIESTA supports multiple parallelization strategies:

1. K-POINT PARALLELIZATION
   - Distributes k-points across MPI processes
   - Best for: Large k-point meshes (metals, accurate calculations)
   - Memory: Each process handles fewer k-points
   - Scaling: Near-linear up to number of k-points

2. BASIS PARALLELIZATION (BlockSize)
   - Distributes orbital operations across processes
   - Best for: Large systems with few k-points
   - Memory: Distributed matrix storage
   - Scaling: Good for large basis sets

3. HYBRID (recommended for most cases)
   - Combine both strategies
   - Balance memory and CPU usage
   - Best for: Medium-to-large systems

Default behavior:
- SIESTA auto-selects based on system size
- Can be overridden with explicit parameters
"""
print(parallelization_overview)


# =============================================================================
# Step 2: K-Point Parallelization
# =============================================================================

print("\n" + "=" * 70)
print("Step 2: K-Point Parallelization")
print("=" * 70)

# Create a test structure
si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# K-point parallel setup
print(
    """
K-point parallelization is optimal when:
- You have many k-points (e.g., 6x6x6 = 216 k-points)
- System has small-to-medium number of atoms
- Memory per process is not a bottleneck

Example setup:
"""
)

# Example: 6x6x6 k-mesh with k-point parallelization
kpoint_parallel_maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    user_params={
        "a2s_kpts": [6, 6, 6],  # 216 irreducible k-points (less with symmetry)
        # SIESTA automatically uses k-point parallelization when beneficial
    },
)

job = kpoint_parallel_maker.make(si)
print(f"Created job: {job.name}")

kpoint_tips = """
Best practices for k-point parallelization:
-------------------------------------------
1. Number of MPI processes should divide number of k-points evenly
   - 216 k-points: use 2, 3, 4, 6, 8, 9, 12, 18, 24, 27, 36, 54, 72, 108, 216 processes

2. Use symmetry to reduce k-points (enabled by default)
   - Si diamond: 216 -> ~20 irreducible k-points

3. For metals, use denser k-mesh but fewer processes per k-point
   - Metal with 12x12x12 mesh: Use all processes for k-points

4. Monitor efficiency with:
   grep "Timer:" job_*/siesta.out
"""
print(kpoint_tips)


# =============================================================================
# Step 3: Basis Parallelization
# =============================================================================

print("\n" + "=" * 70)
print("Step 3: Basis Parallelization")
print("=" * 70)

print(
    """
Basis parallelization is optimal when:
- System has many atoms (>100)
- K-point mesh is small (Gamma-only or 2x2x2)
- You need distributed memory for large matrices

Example setup for large system:
"""
)

# Example: Large supercell with Gamma-only
large_system_maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    user_params={
        "a2s_kpts": [1, 1, 1],  # Gamma-only for large supercell
        "BlockSize": 64,  # Tune based on system size
        # Good values: 16, 32, 64, 128
    },
)

print(
    """
BlockSize parameter:
--------------------
- Controls the block cyclic distribution of matrices
- Smaller BlockSize = more parallelization overhead
- Larger BlockSize = better cache performance, less parallelization

Recommended values:
- Small systems (< 50 atoms): 16-32
- Medium systems (50-200 atoms): 32-64
- Large systems (> 200 atoms): 64-128

Syntax:
  user_params = {"BlockSize": 64}
"""
)


# =============================================================================
# Step 4: Memory Optimization
# =============================================================================

print("\n" + "=" * 70)
print("Step 4: Memory Optimization")
print("=" * 70)

memory_optimization = """
Memory-saving options for large systems:

1. Direct Inversion (for small-gap systems)
   - Avoids full matrix diagonalization
   - Use: "SolutionMethod": "OrderN"  # For large gapped systems

2. Reduce saved data
   user_params = {
       "DM.UseSaveDM": True,      # Save DM for restart (small)
       "SaveHS": False,            # Don't save H/S matrices (large!)
       "WriteWaveFunctions": False, # Don't save WF (very large!)
       "WriteDM": False,           # Don't write DM each step
       "WriteDMHS.NetCDF": False,  # No NetCDF output
   }

3. Reduce real-space grid memory
   user_params = {
       "Mesh.Cutoff": "200 Ry",  # Lower cutoff = less memory
       # But verify convergence first!
   }

4. Sparse matrix options (advanced)
   user_params = {
       "DM.Tolerance": "1e-4",  # Can use 1e-3 for large systems
   }
"""
print(memory_optimization)

# Memory-efficient maker
memory_efficient_maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    user_params={
        "DM.UseSaveDM": True,
        "SaveHS": False,
        "WriteWaveFunctions": False,
    },
)

print("Created memory-efficient maker with minimal output files.")


# =============================================================================
# Step 5: HPC Resource Allocation
# =============================================================================

print("\n" + "=" * 70)
print("Step 5: HPC Resource Allocation")
print("=" * 70)

hpc_allocation = """
General guidelines for HPC resource allocation:

NODES AND CORES
---------------
System Size    | Recommended Cores | Memory/Core
-------------- | ----------------- | -----------
< 20 atoms     | 4-16              | 2 GB
20-100 atoms   | 16-64             | 4 GB
100-500 atoms  | 64-256            | 4-8 GB
> 500 atoms    | 256+              | 8+ GB

WALL TIME ESTIMATES
-------------------
Calculation Type    | Time Factor
------------------- | -----------
Static (SCF)        | 1x base
Relaxation (10 steps)| 5-15x base
NEB (5 images)      | 50-100x base
Phonon (20 displacements) | 20-40x base

JOBFLOW-REMOTE RESOURCE SETTINGS
--------------------------------
# In Python before submission:
from atomate2.siesta.powerups import update_jobflow_resources

job = maker.make(structure)
job = update_jobflow_resources(
    job,
    nodes=2,
    tasks_per_node=48,
    time="12:00:00",
    mem_per_cpu="4GB",
)

# Or via command line after submission:
jf -p PROJECT job set resources <db_id> --nodes 2 --time "24:00:00"
"""
print(hpc_allocation)


# =============================================================================
# Step 6: Performance Benchmarking
# =============================================================================

print("\n" + "=" * 70)
print("Step 6: Performance Benchmarking")
print("=" * 70)

benchmarking = """
How to benchmark your calculations:

1. Run scaling tests:
   - Same calculation with 1, 2, 4, 8, 16 cores
   - Plot time vs cores to find efficiency drop-off

2. Check SIESTA timing:
   grep "Timer:" job_*/siesta.out

   Key sections:
   - siesta: Total = XX.XX s  (total time)
   - compute_dm: Total = XX.XX s  (SCF time)
   - cellxc: Total = XX.XX s  (XC potential)
   - nlefsm: Total = XX.XX s  (Non-local forces)

3. Identify bottlenecks:
   - If compute_dm dominates: k-point or basis bottleneck
   - If cellxc dominates: mesh cutoff too high
   - If nlefsm dominates: pseudopotential operations

4. Efficiency calculation:
   Efficiency = (T_1 / (N * T_N)) * 100%

   Where:
   - T_1 = time with 1 core
   - T_N = time with N cores
   - N = number of cores

   Target: >70% efficiency

Example benchmark script:
"""

benchmark_script = """
#!/bin/bash
# benchmark.sh - Run scaling test

STRUCTURE="si.cif"
CORES_LIST="1 2 4 8 16"

for CORES in $CORES_LIST; do
    echo "Running with $CORES cores..."
    mpirun -np $CORES siesta < siesta.fdf > siesta_${CORES}cores.out 2>&1

    # Extract timing
    TIME=$(grep "siesta: Total =" siesta_${CORES}cores.out | awk '{print $4}')
    echo "$CORES cores: $TIME seconds"
done
"""
print(benchmark_script)


# =============================================================================
# Step 7: Quick Optimization Checklist
# =============================================================================

print("\n" + "=" * 70)
print("Step 7: Quick Optimization Checklist")
print("=" * 70)

checklist = """
PERFORMANCE OPTIMIZATION CHECKLIST
==================================

Before running:
[ ] Choose appropriate k-points (converged but not excessive)
[ ] Set Mesh.Cutoff from convergence test
[ ] Use appropriate basis size (DZP usually sufficient)
[ ] Enable symmetry (default)

For large systems (>100 atoms):
[ ] Consider Gamma-only or sparse k-mesh
[ ] Use BlockSize parameter
[ ] Enable memory-saving options
[ ] Consider OrderN method for large gapped systems

For HPC:
[ ] Estimate memory requirements (4-8 GB/core for large systems)
[ ] Choose cores to match k-point count (k-point parallel)
[ ] Set appropriate wall time
[ ] Request whole nodes when possible

Monitoring:
[ ] Check timer output for bottlenecks
[ ] Verify efficiency > 70%
[ ] Watch for memory issues (OOM killer)

Post-calculation:
[ ] Record optimal parameters for similar systems
[ ] Save timing data for future reference
"""
print(checklist)


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print(
    """
Key optimization strategies:

1. K-point parallelization
   - Best for: metals, accurate calculations
   - Match MPI processes to k-point count

2. Basis parallelization (BlockSize)
   - Best for: large supercells, Gamma-only
   - Adjust BlockSize (16-128)

3. Memory optimization
   - Disable unnecessary output files
   - Use appropriate tolerances

4. HPC resource allocation
   - Scale cores with system size
   - Monitor efficiency

5. Benchmarking
   - Always run scaling tests for production
   - Target >70% parallel efficiency

Next tutorial:
- 02_computational_cost_reduction.py - Reducing calculation cost
"""
)
