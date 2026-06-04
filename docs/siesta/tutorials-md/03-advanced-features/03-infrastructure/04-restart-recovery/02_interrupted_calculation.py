#!/usr/bin/env python
"""Recovery from interrupted calculation (walltime, crash, etc.).

Demonstrates how to recover from an interrupted calculation by:
1. Detecting the interruption
2. Finding the output directory
3. Restarting with prev_dir to continue from last converged step
"""

from pathlib import Path

from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.core import RelaxMaker

# Create a simple Si structure
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

print("=" * 70)
print("Restart Tutorial - Interrupted Calculation Recovery")
print("=" * 70)

# ============================================================================
# Simulating an interrupted calculation
# ============================================================================

print("\n[Scenario] Calculation interrupted due to:")
print("  - Walltime limit on HPC cluster")
print("  - System crash or power failure")
print("  - Manual termination (Ctrl+C)")
print("  - Network issues")

print("\n[Recovery Steps]")
print("-" * 70)

# ============================================================================
# Step 1: Identify the interrupted calculation directory
# ============================================================================

print("\n[Step 1] Find the calculation directory")
print(
    """
Common locations:
  - Local: ./job_<timestamp>_<uuid>
  - HPC: /scratch/$USER/job_<timestamp>_<uuid>
  - Jobflow: Check database for dir_name field

Commands to find it:
  ls -lt job_*              # Find latest job directory
  find . -name "siesta.out" # Find all SIESTA output files
  grep -l "interrupted" job_*/siesta.out  # Find interrupted runs
"""
)

# For this example, we'll create a mock interrupted calculation
print("\n(Creating mock interrupted calculation for demonstration...)")

maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "MD.NumCGsteps": 50,  # Many relaxation steps
    }
)
job_initial = maker.make(structure)

# Run for demonstration
results_initial = run_locally(job_initial, create_folders=True, ensure_success=True)

# Find the directory (use absolute path)
# Note: jobflow stores outputs directly in job_* directory
calc_dirs = sorted(Path(".").glob("job_*"), key=lambda p: p.stat().st_mtime)
interrupted_dir = calc_dirs[-1].resolve()

print(f"\nFound interrupted calculation: {interrupted_dir}")

# ============================================================================
# Step 2: Check what was completed
# ============================================================================

print("\n[Step 2] Check calculation status")
print(f"Examining: {interrupted_dir}")
print(
    """
Files to check:
  siesta.out     - Last SCF iteration completed
  siesta.DM      - Density matrix saved
  siesta.XV      - Last geometry saved
  siesta.STRUCT_OUT - Final structure (if completed)

Look for in siesta.out:
  "SCF cycle converged after N iterations"  - Good!
  "Geometry step N"                         - Progress indicator
  Last line shows where it stopped
"""
)

# Check if key files exist
dm_file = interrupted_dir / "siesta.DM"
xv_file = interrupted_dir / "siesta.XV"
out_file = interrupted_dir / "siesta.out"

print("\nKey files present:")
print(f"  siesta.DM:  {dm_file.exists()}")
print(f"  siesta.XV:  {xv_file.exists()}")
print(f"  siesta.out: {out_file.exists()}")

# ============================================================================
# Step 3: Restart the calculation
# ============================================================================

print("\n[Step 3] Restart from interrupted state")
print("-" * 70)

maker_restart = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "MD.NumCGsteps": 50,  # Same as before
    }
)

# Create restart job using prev_dir
print(f"Creating restart job with prev_dir={interrupted_dir}")
job_restart = maker_restart.make(structure, prev_dir=str(interrupted_dir))

print("\nRestarting calculation...")
print("This will:")
print("  ✓ Load converged density matrix from siesta.DM")
print("  ✓ Continue from last geometry in siesta.XV")
print("  ✓ Complete remaining relaxation steps")
print("  ✓ Generate new output in fresh job directory")

results_restart = run_locally(job_restart, create_folders=True, ensure_success=True)

print("\n✓ Calculation successfully resumed and completed!")

# ============================================================================
# Step 4: Verify completion
# ============================================================================

print("\n[Step 4] Verify completion")
print("-" * 70)

# Find the restart directory
restart_dirs = sorted(
    Path(".").glob("job_*"), key=lambda p: p.stat().st_mtime, reverse=True
)
restart_dir = restart_dirs[0]

print(f"\nRestart job directory: {restart_dir}")
print("\nCheck for completion:")
print(f"  ls {restart_dir}/outputs/siesta.STRUCT_OUT")
print(f"  tail {restart_dir}/outputs/siesta.out")
print("\nLook for final messages:")
print('  "Job completed"')
print('  "Geometry step converged"')
print("  Final energy and forces printed")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(
    """
Recovery workflow:
1. Find interrupted calculation directory
2. Check siesta.out for last completed step
3. Verify siesta.DM and siesta.XV files exist
4. Create new job with prev_dir pointing to outputs/
5. Run the restart job - continues from last state

Benefits:
- No wasted computation - continues from where it stopped
- Faster convergence - reuses converged density matrix
- Preserves progress - no need to start from scratch

Prevention tips:
- Set realistic walltime limits (longer than needed)
- Use custodian with WalltimeHandler (see 03-error-handling/)
- Monitor jobs: squeue, qstat, or cluster status tools
- Test with shorter runs before production calculations

Automatic alternative:
See 04-infrastructure/03-error-handling/ for automatic recovery
with custodian WalltimeHandler that handles this automatically!
"""
)
