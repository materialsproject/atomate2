"""
Performance Optimization - Computational Cost Reduction
=======================================================

This tutorial covers strategies to reduce computational cost:
- Choosing appropriate tier presets
- When to use tighter vs looser convergence
- Calculation hierarchy (cheap -> expensive)
- Cost vs accuracy tradeoffs

Category: troubleshooting/performance_optimization
Difficulty: Intermediate
Time: 25 minutes
"""

from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

# =============================================================================
# Step 1: Understanding Computational Cost
# =============================================================================

print("=" * 70)
print("Step 1: Understanding Computational Cost")
print("=" * 70)

cost_factors = """
Main factors affecting computational cost in SIESTA:

1. K-POINT MESH (O(N_k))
   - Cost scales linearly with number of k-points
   - 6x6x6 is ~27x more expensive than 2x2x2

2. MESH CUTOFF (O(N_grid))
   - Higher cutoff = finer real-space grid
   - 400 Ry is ~8x more expensive than 200 Ry (in 3D)

3. BASIS SIZE (O(N_orb^3))
   - More orbitals = larger matrices
   - TZP is ~3-5x more expensive than DZP

4. SYSTEM SIZE (O(N_atoms^3) for exact diagonalization)
   - Cost scales cubically with atoms
   - 100 atoms is ~8x more expensive than 50 atoms

5. NUMBER OF SCF/OPTIMIZATION STEPS
   - More iterations = longer runtime
   - Tighter tolerance = more iterations

Cost hierarchy (rough estimates):
- Static calculation: 1x
- Relaxation (10 steps): 5-15x
- Band structure: 2-5x
- Phonon (20 displacements): 20-40x
- NEB (5 images x 20 steps): 100-200x
"""
print(cost_factors)


# =============================================================================
# Step 2: Tier Presets for Cost Control
# =============================================================================

print("\n" + "=" * 70)
print("Step 2: Tier Presets for Cost Control")
print("=" * 70)

# Create test structure
si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

tier_comparison = """
Tier presets provide optimized parameters at different cost/accuracy levels:

TIER         | k-points | Mesh Cutoff | Basis | Use Case
-------------|----------|-------------|-------|-------------------
basic_dirty  | 2x2x2    | 150 Ry      | SZ    | Quick tests, screening
basic        | 4x4x4    | 200 Ry      | DZP   | Initial structures
intermediate | 6x6x6    | 300 Ry      | DZP   | Production calculations
advanced     | 8x8x8    | 400 Ry      | TZP   | High accuracy
expert       | 10x10x10 | 500 Ry      | TZP+  | Publication quality

Relative cost (approximate):
- basic_dirty: 1x
- basic: 5x
- intermediate: 20x
- advanced: 100x
- expert: 500x
"""
print(tier_comparison)

# Example: Using tier presets
print("\nExample makers with different tiers:\n")

# Basic dirty - fastest
dirty_maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
dirty_maker = apply_tier_preset(dirty_maker, "relax_dirty")
print("Created: basic_dirty tier maker (fastest, screening)")

# Basic - initial calculations
basic_maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
basic_maker = apply_tier_preset(basic_maker, "relax_standard")
print("Created: basic tier maker (production)")

# Advanced - high accuracy
advanced_maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
advanced_maker = apply_tier_preset(advanced_maker, "relax_high_accuracy")
print("Created: advanced tier maker (high accuracy)")


# =============================================================================
# Step 3: When to Use Tighter vs Looser Convergence
# =============================================================================

print("\n" + "=" * 70)
print("Step 3: When to Use Tighter vs Looser Convergence")
print("=" * 70)

convergence_guide = """
CONVERGENCE REQUIREMENTS BY APPLICATION
=======================================

Application              | SCF Tol | Force Tol | Mesh   | k-mesh
-------------------------|---------|-----------|--------|--------
Initial screening        | 1e-3    | 0.1 eV/A  | 150 Ry | 2x2x2
Structure optimization   | 1e-4    | 0.04 eV/A | 200 Ry | 4x4x4
Band structure           | 1e-4    | N/A       | 250 Ry | 8x8x8
Total energies (EOS)     | 1e-5    | 0.02 eV/A | 300 Ry | 6x6x6
Force constants (phonon) | 1e-6    | 0.01 eV/A | 300 Ry | 4x4x4
Elastic constants        | 1e-5    | 0.01 eV/A | 350 Ry | 8x8x8
NEB barriers             | 1e-4    | 0.04 eV/A | 250 Ry | 4x4x4

Key insight:
- Forces require tighter convergence than energies
- Phonons require tightest convergence (second derivatives)
- Energy differences can use moderate convergence
- Band structure needs dense k-mesh, not tight SCF
"""
print(convergence_guide)


# =============================================================================
# Step 4: Calculation Hierarchy Strategy
# =============================================================================

print("\n" + "=" * 70)
print("Step 4: Calculation Hierarchy Strategy")
print("=" * 70)

hierarchy_strategy = """
RECOMMENDED CALCULATION WORKFLOW
================================

For a new material, follow this hierarchy (cheap -> expensive):

STAGE 1: Quick Screening (1 minute)
-----------------------------------
- tier: basic_dirty
- Purpose: Verify structure is reasonable
- Check: Does SCF converge? No crazy forces?

STAGE 2: Convergence Tests (10-30 minutes)
------------------------------------------
- Test mesh cutoff: [150, 200, 250, 300] Ry
- Test k-points: [[2,2,2], [4,4,4], [6,6,6]]
- Purpose: Find converged parameters
- Invest time here to save time later!

STAGE 3: Structure Relaxation (1-4 hours)
-----------------------------------------
- tier: basic or intermediate
- Use converged parameters from Stage 2
- Purpose: Get accurate equilibrium structure

STAGE 4: Property Calculations (1-24 hours)
-------------------------------------------
- Band structure: Dense k-path, single SCF
- DOS: Dense k-mesh, single SCF
- Phonons: Multiple displacements
- Elastic: Multiple strains

STAGE 5: High-Accuracy Refinement (optional)
-------------------------------------------
- tier: advanced or expert
- Only if needed for publication
- Use for energy differences, band gaps

Cost savings:
- Following hierarchy saves 70-90% compute time vs.
  running everything at high accuracy from the start
"""
print(hierarchy_strategy)

# Example implementation
print("\nExample: Implementing the hierarchy\n")

# Stage 1: Screening
screening_maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    user_params={
        "PAO.BasisSize": "SZ",  # Minimal basis
        "a2s_kpts": [2, 2, 2],
        "Mesh.Cutoff": "150 Ry",
        "MaxSCFIterations": 50,  # Quick fail if problematic
    },
)
print("Stage 1 (Screening):")
print("  - Basis: SZ, k-points: 2x2x2, Mesh: 150 Ry")
print("  - Expected time: ~1 minute")

# Stage 3: Production
production_maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    user_params={
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [6, 6, 6],  # From convergence test
        "Mesh.Cutoff": "250 Ry",  # From convergence test
    },
)
print("\nStage 3 (Production):")
print("  - Basis: DZP, k-points: 6x6x6, Mesh: 250 Ry")
print("  - Expected time: ~1-4 hours")


# =============================================================================
# Step 5: Cost vs Accuracy Tradeoffs
# =============================================================================

print("\n" + "=" * 70)
print("Step 5: Cost vs Accuracy Tradeoffs")
print("=" * 70)

tradeoffs = """
PARAMETER TRADEOFFS
===================

K-POINTS
--------
- Too few: Wrong energies, forces, band structure
- Too many: Wasted compute time
- Sweet spot: Converge total energy to 1 meV/atom

MESH CUTOFF
-----------
- Too low: Egg-box effect, wrong forces
- Too high: Large memory, slow grid operations
- Sweet spot: Converge energy to 1 meV/atom (usually 200-300 Ry)

BASIS SIZE
----------
- SZ: Screening only, qualitatively wrong
- DZ: Missing polarization, incorrect band gaps
- DZP: Standard production (recommended default)
- TZP: High accuracy, 3-5x cost of DZP
- TZP+: Diminishing returns for most systems

SCF TOLERANCE
-------------
- 1e-3: Very fast, energies uncertain by ~10 meV
- 1e-4: Standard, energies uncertain by ~1 meV
- 1e-5: Forces accurate to ~0.01 eV/A
- 1e-6: Phonon-quality forces

FORCE TOLERANCE (geometry optimization)
---------------------------------------
- 0.1 eV/A: Quick optimization, approximate structure
- 0.04 eV/A: Standard production (default)
- 0.02 eV/A: Tight optimization
- 0.01 eV/A: Phonon-ready structure, slow

WHEN ACCURACY MATTERS MOST
--------------------------
Property                | Critical Parameters
------------------------|--------------------
Total energy            | k-points, mesh cutoff
Forces                  | SCF tolerance, mesh cutoff
Band gap                | Basis size, XC functional
Phonon frequencies      | Force tolerance, supercell size
Formation energies      | Consistent settings across all calculations
"""
print(tradeoffs)


# =============================================================================
# Step 6: Practical Cost Reduction Tips
# =============================================================================

print("\n" + "=" * 70)
print("Step 6: Practical Cost Reduction Tips")
print("=" * 70)

practical_tips = """
PRACTICAL COST REDUCTION STRATEGIES
===================================

1. USE SYMMETRY (enabled by default)
   - Reduces k-points by factors of 2-48
   - Free speedup, no accuracy loss
   - Warning: Breaking symmetry (defects) needs care

2. START FROM GOOD STRUCTURE
   - Literature lattice parameters
   - Pre-relaxed with lower accuracy
   - Avoids long optimization paths

3. USE RESTART FILES
   - SaveDM=True for SCF restarts
   - Saves 30-50% time on similar calculations

4. PARALLELIZE INDEPENDENT CALCULATIONS
   - Phonon displacements are independent
   - NEB images (after initial relaxation) are independent
   - EOS volumes are independent

5. USE DRY-RUN FOR TESTING
   - Catches errors before wasting compute time
   - Verify input files are correct

6. BATCH SIMILAR CALCULATIONS
   - Run convergence tests together
   - Reuse k-point and mesh cutoff results

7. CHECKPOINT LONG CALCULATIONS
   - Enable custodian for automatic recovery
   - Use MD.UseSaveXV for geometry restarts

QUANTIFIED SAVINGS
------------------
Strategy              | Time Savings
----------------------|-------------
Symmetry              | 50-90%
Restart from DM       | 30-50%
Hierarchy approach    | 70-90%
Appropriate tier      | 50-high
Parallelization       | Linear with cores (up to limit)
"""
print(practical_tips)


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print(
    """
Key cost reduction strategies:

1. Use tier presets
   - Match accuracy level to application
   - basic_dirty for screening, intermediate for production

2. Follow calculation hierarchy
   - Screening -> Convergence -> Production -> High-accuracy
   - Don't jump to expensive calculations

3. Understand convergence requirements
   - Different properties need different precision
   - Tightest for phonons, loosest for screening

4. Practical optimizations
   - Use symmetry (free speedup)
   - Restart from density matrix
   - Parallelize independent calculations
   - Test with dry-run first

5. Know your tradeoffs
   - DZP is usually sufficient
   - More k-points != always better
   - Converge first, then optimize

Remember: Time spent on convergence tests
saves much more time in production runs!
"""
)
