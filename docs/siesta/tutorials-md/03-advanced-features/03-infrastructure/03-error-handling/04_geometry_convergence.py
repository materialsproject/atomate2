#!/usr/bin/env python
"""Geometry convergence handler and strict validation modes.

This tutorial demonstrates the four modes of geometry convergence handling:

Mode 1 (Default): Fast/dirty - allow non-converged geometries
Mode 2: Auto-recovery - custodian fixes non-converged geometries
Mode 3: Strict checking - must converge or fail with clear error
Mode 4: Paranoid - custodian + strict validation (guaranteed convergence)

New in Session 145: Addresses "what happens if relaxation doesn't converge?"
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.powerups import update_user_siesta_settings
from atomate2.siesta.sets.tiers import apply_tier_preset

# ============================================================================
# CONFIGURATION
# ============================================================================

# Choose which mode to demonstrate
MODE = 4  # Change to 1, 2, 3, or 4

# Structure
structure = Structure.from_file(
    "../../../../tutorials/00-structures/interface_relaxed.cif"
)

# ============================================================================
# MODE 1: FAST/DIRTY (Default - Lenient)
# ============================================================================

if MODE == 1:
    print("=" * 80)
    print("MODE 1: Fast/Dirty Calculation")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - use_custodian=False")
    print("  - strict_convergence=False (default)")
    print("  - Tier preset: 2d_vdw_dirty (few MD steps, SCF.MustConverge=False)")
    print("  - MD.NumCGsteps: 5 (very few steps, won't converge)")
    print()
    print("Behavior:")
    print("  - Non-converged geometries are OK")
    print("  - Calculation continues even if forces above tolerance")
    print("  - No error or warning")
    print()
    print("Use case:")
    print("  - Quick screening of many structures")
    print("  - Testing workflows")
    print("  - Pre-relaxation before fine relaxation")
    print("  - When speed > accuracy")
    print()

    # Create maker with lenient settings and dirty preset
    maker = RelaxMaker.fixed_cell_relaxation(
        # dry_run=True,
        use_custodian=False,  # No error recovery
        strict_convergence=False,  # Allow non-converged (default)
    )

    # Apply dirty tier preset (few MD steps, likely won't converge)
    maker = apply_tier_preset(maker, "2d_vdw_dirty")

    # Make job and override MD steps to ensure non-convergence
    job = maker.make(structure)
    job = update_user_siesta_settings(job, {"MD.NumCGsteps": 5})  # Very few steps

    results = run_locally(job, create_folders=True)

    print("✓ Calculation complete")
    print()
    print("Result: Job succeeded even if not converged")
    print("Check: output.geometry_converged, output.final_max_force in results")
    print()

# ============================================================================
# MODE 2: AUTO-RECOVERY (Production - Custodian)
# ============================================================================

elif MODE == 2:
    print("=" * 80)
    print("MODE 2: Auto-Recovery with Custodian")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - use_custodian=True")
    print("  - strict_convergence=False (default)")
    print("  - Tier preset: 2d_vdw_dirty (few MD steps, likely won't converge)")
    print("  - MD.NumCGsteps: 5 (will trigger GeometryConvergenceHandler)")
    print()
    print("Behavior:")
    print("  - GeometryConvergenceHandler detects non-converged geometry")
    print("  - Automatically increases MD.NumCGsteps")
    print("  - Progressive strategy: +50% → +100% → +150% → +200% → max 1000")
    print("  - Tries alternative methods: FIRE quenching, Broyden optimizer")
    print("  - Validator is lenient (allows handler to work)")
    print()
    print("Use case:")
    print("  - Production calculations")
    print("  - Unattended workflows")
    print("  - HPC job submission")
    print("  - When robustness matters")
    print()

    # Create maker with custodian enabled and dirty preset
    maker = RelaxMaker.fixed_cell_relaxation(
        # dry_run=True,  # Set to False for real calculation
        use_custodian=True,  # Enable error recovery
        strict_convergence=False,  # Lenient validator (default)
        custodian_max_errors=5,  # Standard
    )

    # Apply dirty tier preset (will trigger handler due to few MD steps)
    maker = apply_tier_preset(maker, "2d_vdw_dirty")

    # Make job and override MD steps to ensure handler is triggered
    job = maker.make(structure)
    job = update_user_siesta_settings(job, {"MD.NumCGsteps": 5})  # Will be corrected

    results = run_locally(job, create_folders=True)

    print("✓ Calculation complete")
    print()
    print("Expected behavior:")
    print("  1. Initial calculation with 5 MD steps fails to converge")
    print("  2. GeometryConvergenceHandler detects non-convergence")
    print("  3. Increases MD.NumCGsteps to 15 (+50%)")
    print("  4. Retries calculation")
    print("  5. If still not converged, keeps increasing (20, 25, etc.)")
    print()
    print("Check: custodian.json for corrections applied")
    print("       cat job_*/custodian.json | jq '.actions'")
    print()

# ============================================================================
# MODE 3: STRICT CHECKING (Production - No Custodian)
# ============================================================================

elif MODE == 3:
    print("=" * 80)
    print("MODE 3: Strict Convergence Checking")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - use_custodian=False")
    print("  - strict_convergence=True")
    print("  - Tier preset: 2d_vdw_dirty (few MD steps, will fail validation)")
    print("  - MD.NumCGsteps: 5 (will trigger ValidationError)")
    print()
    print("Behavior:")
    print("  - RelaxationValidator checks geometry convergence")
    print("  - Fails with ValidationError if not converged")
    print("  - Clear error message with suggestions")
    print("  - User must fix parameters and retry")
    print()
    print("Use case:")
    print("  - Production without auto-recovery")
    print("  - When you want immediate feedback")
    print("  - Testing if convergence possible")
    print("  - When you need guaranteed convergence")
    print()

    # Create maker with strict validation and dirty preset
    maker = RelaxMaker.fixed_cell_relaxation(
        # dry_run=True,  # Set to False for real calculation
        use_custodian=False,  # No auto-recovery
        strict_convergence=True,  # Enforce convergence
    )

    # Apply dirty tier preset (will fail validation due to few MD steps)
    maker = apply_tier_preset(maker, "2d_vdw_dirty")

    # Make job and override MD steps to ensure validation failure
    job = maker.make(structure)
    job = update_user_siesta_settings(job, {"MD.NumCGsteps": 5})  # Will fail

    print("Attempting calculation with strict_convergence=True...")
    print()

    # Run calculation - jobflow catches exceptions internally
    results = run_locally(job, create_folders=True, ensure_success=False)

    # Check if job succeeded or failed
    # Note: Failed jobs may not be in results dict if validation raised error
    if job.uuid not in results:
        print("✗ Validation failed (expected)")
        print()
        print("Error: Job failed during validation")
        print()
        print("Suggestions:")
        print("  1. Increase MD.NumCGsteps: 5 → 200 (or more)")
        print("  2. Loosen force tolerance")
        print("  3. Disable strict_convergence for dirty calculations")
        print()
    elif results[job.uuid][1].error:
        job_result = results[job.uuid][1]
        print("✗ Validation failed (expected)")
        print()
        print("Error message:")
        print(f"  {job_result.error}")
        print()
        print("Suggestions from error:")
        print("  1. Increase MD.NumCGsteps")
        print("  2. Loosen force tolerance")
        print("  3. Disable strict_convergence for dirty calculations")
        print()
        print("To fix, increase MD.NumCGsteps:")
        print("  MD.NumCGsteps: 5 → 200 (or more)")
        print()
    else:
        print("✓ Calculation converged successfully (unexpected!)")
        print("  This shouldn't happen with only 5 MD steps")
        print()

# ============================================================================
# MODE 4: PARANOID (Custodian + Strict Validation)
# ============================================================================

elif MODE == 4:
    print("=" * 80)
    print("MODE 4: Paranoid (Custodian + Strict Validation)")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - use_custodian=True")
    print("  - strict_convergence=True")
    print("  - Tier preset: 2d_vdw_dirty (few MD steps, handler will correct)")
    print("  - MD.NumCGsteps: 5 (handler corrects, validator enforces)")
    print()
    print("Behavior:")
    print("  - GeometryConvergenceHandler tries to fix non-convergence")
    print("  - Up to 5 correction attempts")
    print("  - After handler exhausted, validator enforces quality")
    print("  - If still not converged after max attempts → FAIL")
    print("  - Guaranteed: Either converged OR clear error")
    print()
    print("Use case:")
    print("  - Critical production calculations")
    print("  - Phonon/NEB workflows (require converged geometries)")
    print("  - When convergence is mandatory")
    print("  - Maximum robustness + quality assurance")
    print()

    # Create maker with both custodian and strict validation + dirty preset
    maker = RelaxMaker.fixed_cell_relaxation(
        # dry_run=True,  # Set to False for real calculation
        use_custodian=True,  # Enable auto-recovery
        strict_convergence=True,  # Enforce quality after corrections
        custodian_max_errors=5,  # Standard attempts
    )

    # Apply dirty tier preset (handler will correct, then validator checks)
    maker = apply_tier_preset(maker, "2d_vdw_dirty")

    # Make job and override MD steps (handler will try to fix)
    job = maker.make(structure)
    job = update_user_siesta_settings(job, {"MD.NumCGsteps": 5})  # Handler corrects

    print("Attempting calculation with paranoid mode...")
    print()

    # Run calculation - jobflow catches exceptions internally
    results = run_locally(job, create_folders=True, ensure_success=False)

    # Check if job succeeded or failed
    # Note: Failed jobs may not be in results dict if custodian raised ValidationError
    if job.uuid not in results:
        print("✗ Validation failed even after handler attempts")
        print()
        print("This means:")
        print("  1. GeometryConvergenceHandler tried up to 5 times")
        print("  2. Each time increased MD.NumCGsteps progressively")
        print("  3. Still didn't converge (5 steps too few even with increases)")
        print("  4. Validator enforced quality gate → FAIL")
        print()
        print("Error: Custodian ValidationError raised")
        print()
        print("Possible reasons:")
        print("  - Initial MD.NumCGsteps too low (5 steps)")
        print("  - Structure is difficult to converge")
        print("  - Force tolerance too tight")
        print()
        print("Solutions:")
        print("  - Check structure validity")
        print("  - Start with higher MD.NumCGsteps (e.g., 200)")
        print("  - Loosen force tolerance")
        print("  - Review convergence history in custodian.json")
        print()
    elif results[job.uuid][1].error:
        job_result = results[job.uuid][1]
        print("✗ Validation failed even after handler attempts")
        print()
        print("This means:")
        print("  1. GeometryConvergenceHandler tried 5 times")
        print("  2. Each time increased MD.NumCGsteps (7, 10, 12, 15, etc.)")
        print("  3. Still didn't converge")
        print("  4. Validator enforced quality gate → FAIL")
        print()
        print("Error:")
        print(f"  {job_result.error}")
        print()
        print("Possible reasons:")
        print("  - Structure is unreasonable")
        print("  - Force tolerance too tight")
        print("  - System genuinely very difficult")
        print()
        print("Solutions:")
        print("  - Check structure validity")
        print("  - Start with higher MD.NumCGsteps (e.g., 200)")
        print("  - Loosen force tolerance")
        print("  - Review convergence history in custodian.json")
        print()
    else:
        print("✓ Calculation converged successfully!")
        print()
        print("This means:")
        print("  - Handler successfully corrected non-convergence")
        print("  - Validator confirmed quality")
        print("  - Paranoid mode worked perfectly!")
        print()

else:
    print("ERROR: MODE must be 1, 2, 3, or 4")
    print()
    print("MODE 1: Fast/dirty (default - lenient)")
    print("MODE 2: Auto-recovery (custodian)")
    print("MODE 3: Strict checking (no custodian)")
    print("MODE 4: Paranoid (custodian + strict)")

# ============================================================================
# CONVERGENCE METADATA
# ============================================================================

print()
print("=" * 80)
print("CONVERGENCE METADATA")
print("=" * 80)
print()
print("All modes use the same tier preset: 2d_vdw_dirty + MD.NumCGsteps=5")
print("  - This preset has few MD steps (likely to not converge)")
print("  - MD.NumCGsteps=5 overrides preset to ensure non-convergence")
print("  - Perfect for demonstrating the different convergence handling modes")
print()
print("Regardless of mode, convergence information is always tracked:")
print()
print("output.geometry_converged: bool")
print("  - True if geometry converged")
print("  - False if forces above tolerance")
print()
print("output.final_max_force: float")
print("  - Maximum force on any atom (eV/Ang)")
print("  - Compare to tolerance to assess quality")
print()
print("output.force_tolerance: float")
print("  - Force tolerance used for convergence check (eV/Ang)")
print("  - Default: 0.04 eV/Ang")
print()
print("This metadata enables:")
print("  - Post-hoc analysis of convergence")
print("  - Filtering results by quality")
print("  - Debugging non-converged calculations")
print("  - Statistics on convergence rates")
print()

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

print()
print("=" * 80)
print("MODE COMPARISON SUMMARY")
print("=" * 80)
print()
print("┌────────┬────────────┬───────────────┬─────────────────────────────┐")
print("│ Mode   │ Custodian  │ Strict        │ Behavior                    │")
print("├────────┼────────────┼───────────────┼─────────────────────────────┤")
print("│ 1      │ False      │ False         │ Fast, may not converge      │")
print("│        │            │ (default)     │ Use: screening, testing     │")
print("├────────┼────────────┼───────────────┼─────────────────────────────┤")
print("│ 2      │ True       │ False         │ Auto-recovery, lenient      │")
print("│        │            │ (default)     │ Use: production (standard)  │")
print("├────────┼────────────┼───────────────┼─────────────────────────────┤")
print("│ 3      │ False      │ True          │ Must converge or fail       │")
print("│        │            │               │ Use: immediate feedback     │")
print("├────────┼────────────┼───────────────┼─────────────────────────────┤")
print("│ 4      │ True       │ True          │ Auto-recovery + quality gate│")
print("│        │            │               │ Use: critical calculations  │")
print("└────────┴────────────┴───────────────┴─────────────────────────────┘")
print()

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print()
print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print()
print("Choose your mode based on use case:")
print()
print("🔍 Research/Screening:")
print("   → MODE 1 (default): Fast, dirty OK")
print("   → RelaxMaker()")
print()
print("🏭 Production (Standard):")
print("   → MODE 2: Auto-recovery, robust")
print("   → RelaxMaker(use_custodian=True)")
print()
print("⚡ Production (Immediate Feedback):")
print("   → MODE 3: Strict, clear errors")
print("   → RelaxMaker(strict_convergence=True)")
print()
print("🔒 Critical Calculations:")
print("   → MODE 4: Maximum robustness + quality")
print("   → RelaxMaker(use_custodian=True, strict_convergence=True)")
print()
print("📊 Phonon/NEB Workflows:")
print("   → MODE 4 recommended (need converged geometries)")
print()
print("⚠️  IMPORTANT:")
print("   - Convergence metadata ALWAYS tracked (all modes)")
print("   - Check output.geometry_converged in results")
print("   - Review custodian.json after Mode 2/4")
print()

print("=" * 80)
print("Tutorial complete!")
print()
print("Next steps:")
print("  1. Try all 4 modes (change MODE = 1, 2, 3, 4)")
print("  2. Compare behavior for each mode")
print("  3. Check convergence metadata in results")
print("  4. Review custodian.json for modes 2 and 4")
print("  5. Apply appropriate mode to your workflows")
print("=" * 80)
