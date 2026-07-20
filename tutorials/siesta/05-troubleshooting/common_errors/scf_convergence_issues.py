"""
Troubleshooting Guide: SCF Convergence Issues
Category: troubleshooting
Difficulty: intermediate
Time: 15 minutes (reading)
Prerequisites: Basic SIESTA calculations

Learning Objectives:
1. Identify SCF convergence problems
2. Understand causes of SCF failures
3. Apply systematic fixes
4. Use custodian for automatic recovery

Key Concepts:
- Self-consistent field (SCF) convergence
- Mixer settings and algorithms
- Electronic temperature and smearing
- Charge density initialization
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.custodian import SCFConvergenceHandler

# ==============================================================================
# Problem: SCF Not Converging
# ==============================================================================
print("=" * 80)
print("TROUBLESHOOTING: SCF Convergence Issues")
print("=" * 80)

print(
    """
SYMPTOM:
Calculation fails with message like:
  "SCF not converged after 50 iterations"
  "Density matrix did not converge"
  "Oscillating SCF convergence"

COMMON CAUSES:
1. Too aggressive mixing (default may be too high)
2. Poor initial charge density guess
3. System has small/zero band gap (metal)
4. Magnetic system without spin polarization
5. Strong correlations (transition metals)
6. Inadequate k-point sampling
7. Insufficient mesh cutoff

DIAGNOSTIC STEPS:
1. Check siesta.out for SCF iteration energies
2. Look for oscillating or diverging energies
3. Identify whether system is metallic
4. Check if basis set is appropriate
"""
)

# ==============================================================================
# Solution 1: Adjust Mixer Settings
# ==============================================================================
print("\n" + "=" * 80)
print("Solution 1: Reduce Mixing Weight")
print("=" * 80)

# Create a structure that might have convergence issues
# Example: transition metal oxide
lattice = Lattice.cubic(4.2)
structure = Structure(
    lattice,
    ["Ni", "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5]],
)

# Default parameters (might fail to converge)
maker_default = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "300 Ry",
    }
)

print("Default mixer settings:")
print("  SCF.Mixer.Weight: 0.1 (default)")
print("  SCF.Mixer.History: 5 (default)")
print("\nIf SCF doesn't converge, try reducing mixer weight:")

# Reduced mixer weight
maker_fixed = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "300 Ry",
        "SCF.Mixer.Weight": 0.01,  # Much more conservative
        "SCF.Mixer.History": 8,  # Remember more steps
    }
)

print("\nFixed mixer settings:")
print("  SCF.Mixer.Weight: 0.01 (10x more conservative)")
print("  SCF.Mixer.History: 8 (more memory)")
print("\nRule of thumb:")
print("  - Start with 0.1 (default)")
print("  - If fails, try 0.05, 0.01, 0.005")
print("  - Lower = slower but more stable")

# ==============================================================================
# Solution 2: Enable Electronic Temperature (Smearing)
# ==============================================================================
print("\n" + "=" * 80)
print("Solution 2: Add Electronic Temperature (Metals/Small Gap)")
print("=" * 80)

print(
    """
WHEN TO USE:
- Metallic systems (no band gap)
- Systems with very small gaps (< 0.5 eV)
- Convergence oscillations near Fermi level

HOW IT HELPS:
- Smooths occupation numbers near Fermi level
- Reduces oscillations in band occupancy
- Mimics finite temperature effects
"""
)

# Metallic system (aluminum)
al_structure = Structure(
    Lattice.cubic(4.05),
    ["Al"] * 4,
    [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
)

maker_metal = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "a2s_kpts": [6, 6, 6],
        "Mesh.Cutoff": "300 Ry",
        "ElectronicTemperature": "300 K",  # Room temperature
        "OccupationFunction": "FD",  # Fermi-Dirac
    }
)

print("Settings for metals:")
print("  ElectronicTemperature: 100-500 K")
print("  OccupationFunction: FD (Fermi-Dirac)")
print("  Alternative: MP (Methfessel-Paxton)")
print("\nTypical values:")
print("  - Metals: 100-500 K")
print("  - Difficult metals: up to 1000 K")
print("  - Insulators: Not needed (use 0 K)")

# ==============================================================================
# Solution 3: Enable Spin Polarization
# ==============================================================================
print("\n" + "=" * 80)
print("Solution 3: Enable Spin Polarization (Magnetic Systems)")
print("=" * 80)

print(
    """
SYMPTOMS INDICATING MAGNETIC ISSUES:
- Transition metal elements (Fe, Co, Ni, Mn, etc.)
- Lanthanides or actinides
- SCF oscillates but doesn't diverge
- Known magnetic material

SOLUTION: Enable spin polarization
"""
)

# Magnetic system (bcc Fe)
fe_structure = Structure(
    Lattice.cubic(2.87),
    ["Fe"],
    [[0, 0, 0]],
)

maker_magnetic = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "a2s_kpts": [8, 8, 8],
        "Mesh.Cutoff": "300 Ry",
        "Spin": "polarized",
        "%block DM.InitSpin": ["1  +  # Initialize Fe with spin up"],
        "SCF.Mixer.Weight": 0.01,  # Often need slower mixing
    }
)

print("Magnetic system settings:")
print("  Spin: 'polarized'")
print("  DM.InitSpin: Specify initial magnetic moments")
print("  SCF.Mixer.Weight: Often needs to be reduced")
print("\nElement-specific typical moments:")
print("  Fe: 2-3 μB")
print("  Co: 1-2 μB")
print("  Ni: 0.5-1 μB")
print("  Mn: 3-5 μB")

# ==============================================================================
# Solution 4: Improve Initial Guess
# ==============================================================================
print("\n" + "=" * 80)
print("Solution 4: Better Initial Density Matrix")
print("=" * 80)

print(
    """
OPTIONS FOR INITIAL DENSITY:

1. Harris Functional (default):
   - Quick initial guess
   - Works for most cases
   - May fail for complex systems

2. Save Density Matrix for Restart:
   UseSaveData: true
   - Enables automatic restart from DM file
   - Good for continuation calculations

3. Restart from Previous Calculation:
   - Copy DM file from similar structure
   - SIESTA automatically uses if present
   - Best for similar structures

4. Diagonalization Method:
   SolutionMethod: diagon
   - More expensive but can be more stable
   - Better for difficult SCF convergence
"""
)

maker_better_init = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "300 Ry",
        "UseSaveData": True,  # Enable restart capability
        "DM.Tolerance": 1.0e-4,
    }
)

print("Better initialization settings added above.")

# ==============================================================================
# Solution 5: Increase k-Points and Mesh Cutoff
# ==============================================================================
print("\n" + "=" * 80)
print("Solution 5: Improve Numerical Accuracy")
print("=" * 80)

print(
    """
CONVERGENCE ISSUES FROM INSUFFICIENT ACCURACY:

Symptoms:
- SCF "converges" but energies oscillate
- Different results with slight structure changes
- Metallizes artificially

Fix: Increase k-points and/or mesh cutoff
"""
)

maker_high_accuracy = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "a2s_kpts": [8, 8, 8],  # Dense k-mesh
        "Mesh.Cutoff": "400 Ry",  # High cutoff
        "PAO.EnergyShift": "50 meV",  # Tighter basis
        "SCF.Mixer.Weight": 0.02,
    }
)

print("High-accuracy settings:")
print("  kpts: [8, 8, 8] (denser than typical 4×4×4)")
print("  Mesh.Cutoff: 400 Ry (vs typical 250-300)")
print("  PAO.EnergyShift: 50 meV (tighter basis)")
print("\nConvergence testing checklist:")
print("  ☐ Test k-mesh: 2×2×2, 4×4×4, 6×6×6, 8×8×8")
print("  ☐ Test cutoff: 200, 250, 300, 350, 400 Ry")
print("  ☐ Energy converged to < 0.01 eV/atom")

# ==============================================================================
# Solution 6: Use Custodian for Automatic Fixing
# ==============================================================================
print("\n" + "=" * 80)
print("Solution 6: Automatic Error Recovery with Custodian")
print("=" * 80)

print(
    """
CUSTODIAN ADVANTAGES:
- Automatically detects SCF convergence failures
- Applies fixes progressively
- Retries with adjusted parameters
- No manual intervention needed

SCF CONVERGENCE RESCUE STRATEGY:
1. Reduce mixer weight (0.1 → 0.05)
2. Further reduce (0.05 → 0.01)
3. Add electronic temperature if metallic
4. Reduce even more (0.01 → 0.005)
5. Final attempt with very conservative settings

Success rate: typically with custodian vs many without
"""
)

# Enable custodian with SCF handler
maker_with_custodian = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "300 Ry",
    },
    use_custodian=True,  # Enable automatic error handling!
    custodian_handlers=[SCFConvergenceHandler(max_attempts=5)],
    custodian_max_errors=10,
)

print("Custodian-enabled maker created.")
print("If SCF fails, custodian will:")
print("  1. Detect the failure")
print("  2. Modify parameters automatically")
print("  3. Restart calculation")
print("  4. Repeat up to 5 times")
print("\nRecommended: ALWAYS use custodian for production runs!")

# ==============================================================================
# Diagnostic Workflow
# ==============================================================================
print("\n" + "=" * 80)
print("Step-by-Step Diagnostic Workflow")
print("=" * 80)

print(
    """
STEP 1: Identify the Problem
□ Check siesta.out for error messages
□ Look at SCF iteration history
□ Note if energy oscillates, diverges, or slowly converges

STEP 2: Quick Fixes (try in order)
□ Reduce mixer weight: 0.1 → 0.05 → 0.01
□ Increase mixer history: 5 → 8 → 12
□ Increase max SCF iterations: 50 → 100 → 200

STEP 3: System-Specific Fixes
□ Metallic system → Add electronic temperature
□ Magnetic elements → Enable spin polarization
□ Small system → Check if you need more atoms
□ Heavy elements → Consider including SOC

STEP 4: Accuracy Issues
□ Increase k-point density
□ Increase mesh cutoff
□ Improve basis set (DZ → DZP → TZP)

STEP 5: Advanced Fixes
□ Change mixer algorithm (Pulay → Broyden)
□ Adjust density matrix tolerance
□ Try different occupation functions
□ Use divide-and-conquer diagonalization

STEP 6: Enable Custodian
□ use_custodian=True for automatic handling
□ Let custodian try progressive fixes
□ high success rate with custodian

STEP 7: If All Else Fails
□ Check structure for issues (atoms too close, etc.)
□ Try with different DFT code to verify structure
□ Consult SIESTA mailing list
□ Consider if system is actually problematic
"""
)

# ==============================================================================
# Prevention is Better Than Cure
# ==============================================================================
print("\n" + "=" * 80)
print("Preventing SCF Convergence Issues")
print("=" * 80)

print(
    """
BEST PRACTICES TO AVOID SCF PROBLEMS:

1. ALWAYS start with:
   - Tier presets for material type
   - use_custodian=True
   - Reasonable k-points and cutoff

2. For NEW materials:
   - Start with quick_characterization
   - Check if it's metallic/magnetic
   - Then use appropriate settings

3. KNOW your system:
   - Metals → need smearing
   - Magnetic → need spin polarization
   - Transition metals → may need conservative mixing
   - Insulators → usually easier

4. CONVERGENCE testing first:
   - Test k-mesh convergence
   - Test cutoff convergence
   - Don't use unconverged parameters

5. USE tier presets:
   - "metals_standard" → has smearing
   - "magnetic_standard" → has spin
   - "oxides_standard" → appropriate for oxides
   - Saves you from most issues!

6. MONITOR early:
   - Check first few SCF iterations
   - If oscillating early, fix immediately
   - Don't wait for 50 iterations to fail

7. KEEP logs:
   - Save working parameters
   - Document what worked/failed
   - Build your own library of fixes
"""
)

# ==============================================================================
# Example: Complete Troubleshooting Session
# ==============================================================================
print("\n" + "=" * 80)
print("Example: Troubleshooting a Real Calculation")
print("=" * 80)

print(
    """
PROBLEM:
NiO calculation fails to converge after 50 SCF iterations.

STEP 1: Analyze the system
- NiO is an antiferromagnetic insulator
- Contains transition metal (Ni)
- Known to be difficult (strong correlations)

STEP 2: Initial attempt (likely failed):
- Default parameters
- No spin polarization
→ SCF oscillates, doesn't converge

STEP 3: Fix 1 - Enable spin:
- Add: Spin = 'polarized'
- Add: Initial magnetic moments
→ Better, but still oscillates

STEP 4: Fix 2 - Reduce mixing:
- SCF.Mixer.Weight: 0.01
- SCF.Mixer.History: 8
→ Converges slowly, takes 80 iterations

STEP 5: Fix 3 - Add smearing (small gap):
- ElectronicTemperature: 100 K
→ Converges in 40 iterations!

STEP 6: Production calculation:
- Use all fixes above
- Enable custodian
- Increase k-points for accuracy
→ Reliable, reproducible results

LESSON:
Systematic approach finds solution. With custodian,
most of this happens automatically!
"""
)

# ==============================================================================
# Exercises
# ==============================================================================
print("\n" + "=" * 80)
print("Exercises")
print("=" * 80)

print(
    """
1. BASIC: Diagnose an SCF Failure
   - Run a calculation that fails to converge
   - Examine siesta.out
   - Identify oscillating vs diverging behavior
   - Apply mixer weight fix

2. INTERMEDIATE: Systematic Fixes
   - Create a transition metal oxide (e.g., FeO)
   - Start with default parameters (should fail)
   - Add spin polarization
   - Adjust mixer settings
   - Document what works

3. ADVANCED: Custodian vs Manual
   - Run same problematic system two ways:
     a) Manual fixes based on this guide
     b) With custodian enabled
   - Compare computational time
   - Note success rates

4. EXPERT: Build Your SCF Fixer
   - Create a helper function that:
     • Analyzes system (magnetic? metallic?)
     • Suggests appropriate SCF parameters
     • Returns configured maker
   - Test on various materials

5. CHALLENGE: The Impossible System
   - Find the most difficult system you can
   - Apply all strategies from this guide
   - Document your problem-solving process
   - Share solution with community

Further Reading:
- SIESTA manual: SCF convergence section
- Custodian documentation: docs/source/custodian.rst
- Tutorial: 04-infrastructure/15-custodian-error-handling
- SIESTA wiki: Common convergence problems
"""
)
