#!/usr/bin/env python
"""Basic restart example - reusing converged density matrix.

This demonstrates how to restart a calculation from a previous run,
reusing the converged density matrix for faster SCF convergence.
"""

import re
from pathlib import Path

from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.core import RelaxMaker

# Create a simple Si structure
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

print("=" * 70)
print("Restart Tutorial - Basic Example")
print("=" * 70)

# ============================================================================
# Step 1: Run initial calculation
# ============================================================================

print("\n[Step 1] Running initial calculation...")
print("-" * 70)

maker = RelaxMaker.fixed_cell_relaxation()
job1 = maker.make(structure)

results1 = run_locally(job1, create_folders=True, ensure_success=True)

# Find the output directory (use absolute path)
# Note: jobflow stores outputs directly in job_* directory
calc_dirs = sorted(Path(".").glob("job_*"), key=lambda p: p.stat().st_mtime)
prev_dir = calc_dirs[-1].resolve()

print("\nInitial calculation completed!")
print(f"Output directory: {prev_dir}")

# ============================================================================
# Step 2: Restart from previous calculation
# ============================================================================

print("\n[Step 2] Restarting calculation from previous run...")
print("-" * 70)
print(f"Using prev_dir: {prev_dir}")
print("\nReusing:")
print("  ✓ Converged density matrix (.DM file)")
print("  ✓ Electronic structure")
print("  ✓ Final geometry")
print("\nExpected result: Faster SCF convergence!")

# Create restart job with prev_dir
# IMPORTANT: For TRUE restart with DM file reuse, we need restart_to_input=True
# Note: prev_dir alone only reuses structure/geometry
#       restart_to_input=True copies DM file for faster SCF convergence
maker_restart = RelaxMaker.fixed_cell_relaxation(
    copy_siesta_kwargs={"restart_to_input": True}  # Enable DM file copy
)
job2 = maker_restart.make(structure, prev_dir=str(prev_dir))

results2 = run_locally(job2, create_folders=True, ensure_success=True)

print("\nRestart calculation completed!")

# ============================================================================
# Step 3: Compare SCF iterations
# ============================================================================

print("\n[Step 3] Comparison")
print("-" * 70)

# Find the job directories
calc_dirs = sorted(Path(".").glob("job_*"), key=lambda p: p.stat().st_mtime)
initial_dir = calc_dirs[0]
restart_dir = calc_dirs[1]


# Count SCF iterations
def count_scf_iterations(siesta_out):
    """Count SCF iterations from siesta.out"""
    with open(siesta_out) as f:
        content = f.read()
    scf_lines = re.findall(r"^\s*scf:\s+\d+", content, re.MULTILINE)
    return len(scf_lines)


initial_scf = count_scf_iterations(initial_dir / "siesta.out")
restart_scf = count_scf_iterations(restart_dir / "siesta.out")

print("\nSCF iterations:")
print(f"  Initial calculation: {initial_scf} iterations")
print(f"  Restart calculation: {restart_scf} iterations")
print(
    f"  Speedup: {100 * (initial_scf - restart_scf) / initial_scf:.0f}% fewer iterations!"
)
print("\n✓ Restart converged faster due to reused density matrix!")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(
    """
Key points:
1. prev_dir parameter: Tells calculation where previous run is located
2. restart_to_input=True: Actually copies DM file for restart
3. Why both are needed:
   - prev_dir alone: Only reuses structure/geometry (for parameter sweeps)
   - prev_dir + restart_to_input: True restart with DM (faster convergence)

Two use cases:

A. TRUE RESTART (same parameters, continue calculation):
   copy_siesta_kwargs={"restart_to_input": True}
   → Copies DM file, faster SCF convergence
   → Use for: interrupted calculations, refining convergence

B. PARAMETER SWEEP (different parameters, same structure):
   No restart_to_input needed
   → Only reuses structure, NOT DM file
   → Use for: basis/k-point/mesh convergence studies
   → Examples: EOSBasisConvergenceMaker, MeshCutoffConvergenceFlowMaker

Why not always copy DM?
- DM file from different basis set → wrong for new calculation
- DM file from different k-points → incompatible mesh
- DM file from different parameters → could cause convergence issues

Benefits of proper restart:
✓ Faster SCF convergence (fewer iterations)
✓ Reuses converged electronic structure
✓ Saves computation time
✓ Useful for long calculations split across walltime limits

Next steps:
- Try 02_interrupted_calculation.py for walltime recovery
- See 03_manual_recovery.py for fixing failed calculations
- Check 04-infrastructure/03-error-handling/ for automatic recovery
"""
)
