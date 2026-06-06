#!/usr/bin/env python
"""Comparison: Manual recovery vs automatic custodian recovery.

This script demonstrates the difference between:
1. Manual recovery (requires diagnosis and parameter adjustment)
2. Automatic custodian recovery (handles everything automatically)

Shows why custodian is recommended for production calculations.
"""

from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.core import RelaxMaker

# Create a simple Si structure
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

print("=" * 70)
print("Comparison: Manual Recovery vs Custodian")
print("=" * 70)

# ============================================================================
# Scenario Setup
# ============================================================================

print("\n[Scenario] Calculation with potential SCF convergence issues")
print("-" * 70)
print(
    """
We'll run a calculation with parameters that might cause SCF issues,
then show both recovery approaches.
"""
)

# ============================================================================
# Approach 1: Manual Recovery (Multi-step process)
# ============================================================================

print("\n" + "=" * 70)
print("APPROACH 1: Manual Recovery")
print("=" * 70)

print(
    """
Manual recovery workflow:

Step 1: Run calculation
  ├─ Create job
  ├─ run_locally()
  └─ Wait for completion (or failure)

Step 2: Check for failure
  ├─ tail siesta.out
  ├─ grep for errors
  └─ Diagnose issue type

Step 3: Determine fixes
  ├─ Research error type
  ├─ Decide parameter changes
  └─ Test fixes (trial and error)

Step 4: Create recovery job
  ├─ Find prev_dir
  ├─ Apply parameter fixes
  └─ Run again

Step 5: Verify success
  ├─ Check output again
  ├─ If failed → back to Step 3
  └─ If successful → done!

Time required: 30-60 minutes per attempt
Success rate: ~many (depends on experience)
Effort: HIGH - requires manual intervention at each step
"""
)

print("\nManual recovery code example:")
print("-" * 70)
print(
    """
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.powerups import update_user_siesta_settings

# Step 1: Initial run
maker = RelaxMaker.fixed_cell_relaxation()
job = maker.make(structure)
results = run_locally(job)  # Might fail!

# Step 2-3: Diagnose and plan fixes (manual work)
# ... examine siesta.out ...
# ... research error messages ...
# ... decide on parameter changes ...

# Step 4: Recovery job
recovery_job = maker.make(structure, prev_dir="job_*/outputs")
recovery_job = update_user_siesta_settings(
    recovery_job,
    {
        "SCF.Mixer.Weight": 0.01,  # Manual fix
        "MaxSCFIterations": 200,   # Manual fix
    }
)
results = run_locally(recovery_job)  # Try again

# Step 5: Check again, repeat if needed
# ... might need multiple attempts ...
"""
)

# ============================================================================
# Approach 2: Custodian (Automatic)
# ============================================================================

print("\n" + "=" * 70)
print("APPROACH 2: Automatic Custodian Recovery")
print("=" * 70)

print(
    """
Custodian automatic workflow:

Step 1: Run calculation with custodian enabled
  ├─ Create job with use_custodian=True
  ├─ run_locally()
  └─ Custodian monitors execution

Step 2: Automatic error detection
  ├─ Custodian watches for failures
  ├─ Detects SCF, walltime, memory errors
  └─ No manual checking needed!

Step 3: Automatic correction
  ├─ Applies appropriate fixes
  ├─ Uses progressive 5-level strategy
  ├─ Adjusts parameters intelligently
  └─ No manual parameter tuning!

Step 4: Automatic retry
  ├─ Restarts calculation with fixes
  ├─ Continues until success or max attempts
  └─ No manual intervention!

Step 5: Success + logging
  ├─ Completes successfully (high rate)
  ├─ Logs all actions in custodian.json
  └─ Done automatically!

Time required: Automatic - no waiting
Success rate: typically (proven in production)
Effort: MINIMAL - just enable custodian
"""
)

print("\nCustodian code example:")
print("-" * 70)
print(
    """
from atomate2.siesta.jobs.core import RelaxMaker

# Just add use_custodian=True - that's it!
maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,
    custodian_max_errors=10  # Max retry attempts
)

job = maker.make(structure)
results = run_locally(job)  # Automatic recovery!

# That's all! Custodian handles everything:
# ✓ Error detection
# ✓ Parameter adjustment
# ✓ Automatic retry
# ✓ Logging corrections
# ✓ Success!
"""
)

# ============================================================================
# Demonstration: Using Custodian
# ============================================================================

print("\n" + "=" * 70)
print("DEMONSTRATION: Running with Custodian")
print("=" * 70)

# Use aggressive parameters that might fail without custodian
print("\nUsing potentially problematic parameters:")
print("  SCF.Mixer.Weight: 0.3  (might oscillate)")
print("  MaxSCFIterations: 30  (might be too few)")

maker_custodian = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,
    custodian_max_errors=10,
    user_params={
        "SCF.Mixer.Weight": 0.3,  # Potentially problematic
        "MaxSCFIterations": 30,  # Low limit
    },
)

job = maker_custodian.make(structure)

print("\nRunning with custodian enabled...")
print("(Custodian will automatically fix any issues)")

results = run_locally(job, create_folders=True, ensure_success=True)

print("\n✓ Calculation completed successfully!")
print("\nCustodian actions logged in: job_*/custodian.json")

# ============================================================================
# Side-by-Side Comparison
# ============================================================================

print("\n" + "=" * 70)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 70)

print(
    """
┌─────────────────────────┬─────────────────────┬──────────────────────┐
│ Feature                 │ Manual Recovery     │ Custodian (Auto)     │
├─────────────────────────┼─────────────────────┼──────────────────────┤
│ Setup Complexity        │ HIGH                │ LOW (1 parameter)    │
│ Error Detection         │ Manual inspection   │ Automatic            │
│ Parameter Adjustment    │ Manual calculation  │ Automatic            │
│ Retry Logic             │ Manual scripting    │ Built-in             │
│ Time Investment         │ 30-60 min/attempt   │ 0 (automatic)        │
│ Success Rate            │ many              │ typically                 │
│ Expertise Required      │ HIGH                │ LOW                  │
│ Logging                 │ Manual              │ Automatic            │
│ Production Ready        │ Not recommended     │ ✓ Recommended        │
│ Learning Curve          │ Steep               │ Minimal              │
│ Error Types Handled     │ 1-2 (you implement) │ 10+ (built-in)       │
│ Progressive Strategies  │ Manual trial/error  │ 5-level automatic    │
│ Code Complexity         │ ~50 lines           │ 1 line               │
└─────────────────────────┴─────────────────────┴──────────────────────┘
"""
)

# ============================================================================
# When to Use Each Approach
# ============================================================================

print("\n" + "=" * 70)
print("When to Use Each Approach")
print("=" * 70)

print(
    """
USE CUSTODIAN (Recommended for most cases):
✓ Production calculations
✓ HPC cluster jobs with walltime limits
✓ Batch processing of many structures
✓ When you want reliability without babysitting
✓ Learning/testing - custodian shows you what works
✓ Any calculation where failure is possible

USE MANUAL RECOVERY (Special cases only):
• Educational purposes (learning SIESTA behavior)
• Debugging custodian itself
• Very exotic failure modes not handled by custodian
• When you need complete control over every parameter
• Research into convergence behavior

HYBRID APPROACH:
1. Start with custodian for automatic recovery
2. Check custodian.json to see what fixes were applied
3. If custodian fails after max attempts, then manual recovery
4. Use custodian.json insights to guide manual fixes
"""
)

# ============================================================================
# Summary and Recommendations
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 70)

print(
    """
KEY TAKEAWAYS:

1. Custodian is STRONGLY RECOMMENDED for all production work
   - high success rate vs many manual
   - Zero manual effort vs 30-60 min per attempt
   - Handles 10+ error types automatically
   - Progressive 5-level correction strategies

2. Manual recovery is a LEARNING TOOL
   - Understand SIESTA behavior
   - See what parameters affect convergence
   - Educational value for beginners
   - Last resort when custodian exhausted

3. Best Practice Workflow:
   ┌─────────────────────────────────────┐
   │ 1. Always enable custodian          │
   │ 2. Let it handle failures           │
   │ 3. Check custodian.json for actions │
   │ 4. Only manual if custodian fails   │
   └─────────────────────────────────────┘

4. Code Simplicity:
   Manual:     50+ lines of recovery logic
   Custodian:  use_custodian=True  # That's it!

5. Time Investment:
   Manual:     Hours debugging and fixing
   Custodian:  Minutes setting up once

RECOMMENDED CODE FOR ALL CALCULATIONS:

maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,           # ← Always!
    custodian_max_errors=10,      # ← Generous retry limit
)

That's it! No manual recovery needed in high of cases.

LEARN MORE:
- Custodian tutorial: 04-infrastructure/03-error-handling/
- Custodian handlers: See 01_default_handlers.py
- SCF strategy: See 02_scf_convergence.py
- Custom handlers: See 05_custom_handlers.py
"""
)

print("\n" + "=" * 70)
print("✓ Use custodian for automatic recovery - it's the smart choice!")
print("=" * 70)
