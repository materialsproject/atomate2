#!/usr/bin/env python
"""Manual recovery from failed calculation with parameter adjustment.

Demonstrates how to manually recover from a failed calculation by:
1. Diagnosing the failure
2. Adjusting parameters to fix the issue
3. Restarting with both prev_dir and corrected parameters
"""

from pathlib import Path

from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.powerups import update_user_siesta_settings

# Create a simple Si structure
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

print("=" * 70)
print("Restart Tutorial - Manual Recovery with Parameter Adjustment")
print("=" * 70)

# ============================================================================
# Scenario: SCF convergence failure
# ============================================================================

print("\n[Scenario] Common calculation failures:")
print(
    """
1. SCF NOT CONVERGED
   - Symptoms: "SCF did not converge", oscillating energy
   - Fix: Reduce mixer weight, increase max iterations

2. GEOMETRY STUCK
   - Symptoms: Forces not decreasing, oscillating structure
   - Fix: Reduce MD step size, change optimization method

3. NEGATIVE EIGENVALUES
   - Symptoms: "Negative eigenvalue", unstable calculation
   - Fix: Increase k-points, refine basis set

4. MEMORY ERRORS
   - Symptoms: Segmentation fault, allocation failed
   - Fix: Reduce basis size, use DiagMemory setting
"""
)

# ============================================================================
# Step 1: Simulate a problematic calculation
# ============================================================================

print("\n[Step 1] Running calculation with aggressive parameters (may fail)")
print("-" * 70)

# Use aggressive mixer that might cause issues
maker_aggressive = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "SCF.Mixer.Weight": 0.5,  # Too aggressive!
        "MaxSCFIterations": 20,  # Too few!
        "SCF.DM.Tolerance": 1.0e-6,  # Very strict
    }
)

job_initial = maker_aggressive.make(structure)

print("Running with aggressive settings:")
print("  SCF.Mixer.Weight: 0.5    (aggressive mixing)")
print("  MaxSCFIterations: 20    (low limit)")
print("  SCF.DM.Tolerance: 1.0e-6 (very strict)")

# Run and expect potential issues
try:
    results_initial = run_locally(
        job_initial, create_folders=True, ensure_success=False
    )
    # Note: jobflow stores outputs directly in job_* directory
    calc_dirs = sorted(Path(".").glob("job_*"), key=lambda p: p.stat().st_mtime)
    failed_dir = calc_dirs[-1].resolve()
    print(f"\nCalculation directory: {failed_dir}")
except Exception as e:
    print(f"\nNote: Calculation may have issues: {e}")
    # Find the directory anyway
    calc_dirs = sorted(Path(".").glob("job_*"), key=lambda p: p.stat().st_mtime)
    if calc_dirs:
        failed_dir = calc_dirs[-1].resolve()
    else:
        # Fallback - find any job directory
        all_dirs = list(Path(".").glob("job_*"))
        if all_dirs:
            failed_dir = all_dirs[-1].resolve()
        else:
            print("ERROR: No job directories found!")
            import sys

            sys.exit(1)

# ============================================================================
# Step 2: Diagnose the failure
# ============================================================================

print("\n[Step 2] Diagnosing the failure")
print("-" * 70)
print(f"\nExamine: {failed_dir}/siesta.out")
print(
    """
Commands to check:
  tail -100 siesta.out           # Last 100 lines
  grep -i "error" siesta.out     # Find error messages
  grep "scf" siesta.out          # Check SCF progress
  grep -i "converged" siesta.out # Check convergence status

Common failure patterns:

SCF NOT CONVERGED:
  grep "SCF cycle" siesta.out | tail -5
  → Energy oscillating? → Reduce mixer weight

GEOMETRY NOT CONVERGING:
  grep "Geometry step" siesta.out
  → Forces not decreasing? → Adjust MD parameters

OUT OF TIME:
  grep -i "time" siesta.out | tail -3
  → Walltime exceeded? → Use custodian WalltimeHandler
"""
)

# ============================================================================
# Step 3: Prepare recovery parameters
# ============================================================================

print("\n[Step 3] Prepare corrected parameters")
print("-" * 70)

recovery_params = {
    # Fix SCF convergence issues
    "SCF.Mixer.Weight": 0.01,  # Much gentler mixing
    "MaxSCFIterations": 200,  # More iterations
    "SCF.DM.Tolerance": 1.0e-4,  # Relaxed tolerance
    # Additional safety measures
    "SCF.Mix.AfterConvergence": True,  # Extra mixing after convergence
    "DM.UseSaveDM": True,  # Use saved density matrix
}

print("\nCorrected parameters:")
for key, value in recovery_params.items():
    print(f"  {key:30s}: {value}")

print("\nReasoning:")
print("  • Reduced mixer weight: Gentler updates, more stable")
print("  • Increased iterations: More chances to converge")
print("  • Relaxed tolerance: Easier to achieve")
print("  • Mix after convergence: Extra stability")
print("  • Use saved DM: Start from previous progress")

# ============================================================================
# Step 4: Create recovery job
# ============================================================================

print("\n[Step 4] Creating recovery job")
print("-" * 70)

# Create new maker with better parameters
maker_recovery = RelaxMaker.fixed_cell_relaxation()

# Create job with prev_dir to reuse progress
print(f"Setting prev_dir: {failed_dir}")
job_recovery = maker_recovery.make(structure, prev_dir=str(failed_dir))

# Apply corrected parameters
print("Applying corrected parameters...")
job_recovery = update_user_siesta_settings(job_recovery, recovery_params)

print("\nRecovery job configured:")
print("  ✓ Reuses converged density matrix from failed run")
print("  ✓ Continues from last geometry state")
print("  ✓ Uses corrected SCF parameters")
print("  ✓ Better chance of convergence")

# ============================================================================
# Step 5: Run recovery calculation
# ============================================================================

print("\n[Step 5] Running recovery calculation")
print("-" * 70)

results_recovery = run_locally(job_recovery, create_folders=True, ensure_success=True)

print("\n✓ Recovery calculation completed successfully!")

# Find recovery directory (most recent)
recovery_dirs = sorted(
    Path(".").glob("job_*"), key=lambda p: p.stat().st_mtime, reverse=True
)
recovery_dir = recovery_dirs[0].resolve()

print(f"\nRecovery job directory: {recovery_dir}")

# ============================================================================
# Step 6: Verify success
# ============================================================================

print("\n[Step 6] Verify recovery success")
print("-" * 70)

print("\nCheck recovery job output:")
print(f"  cat {recovery_dir}/outputs/siesta.out | grep -A 5 'Final energy'")
print(f"  tail -50 {recovery_dir}/outputs/siesta.out")

print("\nLook for success indicators:")
print("  ✓ 'SCF cycle converged'")
print("  ✓ 'Begin CG opt. move =' messages")
print("  ✓ 'Job completed' at end")
print("  ✓ Final energy and forces printed")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("Summary - Manual Recovery Workflow")
print("=" * 70)
print(
    """
Step-by-step recovery:

1. DIAGNOSE
   - Check siesta.out for error messages
   - Identify failure type (SCF, geometry, memory, etc.)
   - Grep for specific patterns

2. PLAN FIXES
   SCF issues        → Adjust mixer, iterations, tolerance
   Geometry issues   → Change MD parameters, force tolerance
   Memory issues     → Reduce basis, adjust memory settings
   Walltime issues   → Increase time or use checkpointing

3. CREATE RECOVERY JOB
   - Use prev_dir to reuse progress
   - Apply corrected parameters with update_user_siesta_settings()
   - Keep structure and other settings

4. RUN AND VERIFY
   - Monitor output for improvement
   - Check convergence messages
   - Verify final results are reasonable

5. IF STILL FAILS
   - Try more aggressive fixes
   - Consider different approach (basis set, method)
   - Or use automatic custodian recovery!

Common parameter fixes:

SCF convergence:
  SCF.Mixer.Weight: 0.5 → 0.01 (gentler)
  MaxSCFIterations: 50 → 200 (more)
  SCF.DM.Tolerance: 1e-6 → 1e-4 (relaxed)

Geometry optimization:
  MD.MaxForceTol: 0.001 → 0.01 (relaxed)
  MD.MaxStressTol: 0.001 → 0.01 (relaxed)
  MD.UseSaveXV: False → True (checkpointing)

Automatic alternative:
For automatic recovery without manual intervention, use custodian!
See: 04-infrastructure/03-error-handling/

Custodian automatically:
- Detects failures
- Applies appropriate fixes
- Retries calculations
- Logs all corrections
- high success rate!
"""
)
