#!/usr/bin/env python3
"""
Tutorial: Creating Custom Tier Presets
=======================================

Learn how to create your own tier presets that automatically appear in CLI tools.

Important: This tutorial shows the structure but doesn't modify the codebase.
To actually add presets, edit files in:
    src/atomate2/siesta/sets/tiers/presets/

After adding, presets automatically appear in:
    atomate2siesta-presets list
    atomate2siesta-presets show <preset_name>
"""

# ==============================================================================
# PRESET STRUCTURE
# ==============================================================================

print("=" * 50)
print("How to Create a Custom Preset")
print("=" * 50)

print(
    """
A preset is a Python dictionary with 4 required components:

1. tier (required): "basic", "intermediate", "advanced", or "expert"
2. description (required): Clear description of use case
3. categories (required): List of category names
4. params (required): Dictionary of SIESTA parameters

Example structure:
"""
)

example = """
MY_PRESET = {
    "tier": "intermediate",
    "description": "What this preset does and when to use it",
    "categories": ["category1", "category2"],
    "params": {
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "200 Ry",
        # Add more parameters...
    },
}
"""
print(example)

# ==============================================================================
# EXAMPLE 1: MOF Screening Preset
# ==============================================================================

print("\n" + "=" * 50)
print("Example 1: MOF Screening Preset")
print("=" * 50)

print(
    """
File: src/atomate2/siesta/sets/tiers/presets/molecular.py

Add this code:
"""
)

mof_code = """
MOF_SCREENING = {
    "tier": "basic",
    "description": (
        "Fast screening for metal-organic frameworks. "
        "Optimized for large unit cells with Γ-point sampling."
    ),
    "categories": ["molecular", "structural"],
    "params": {
        "PAO.BasisSize": "DZP",
        "Mesh.Cutoff": "200 Ry",
        "a2s_kpts": [1, 1, 1],              # Γ-only for large MOFs
        "SCF.Mixer.Method": "Pulay",
        "SCF.Mixer.Weight": 0.005,      # Slow mixing for stability
        "SCF.Mixer.History": 8,
        "SCF.DM.Tolerance": 1e-4,
        "MD.MaxForceTol": "0.04 eV/Ang",
    },
}
"""
print(mof_code)

print(
    """
Then update __init__.py:

1. Import the preset:
   from .molecular import MOF_SCREENING

2. Add to TIER_PRESETS dict:
   TIER_PRESETS = {
       ...
       "mof_screening": MOF_SCREENING,
   }

Done! Automatically appears in CLI.
"""
)

# ==============================================================================
# EXAMPLE 2: Catalysis Preset with vdW
# ==============================================================================

print("\n" + "=" * 50)
print("Example 2: Catalysis Preset with vdW")
print("=" * 50)

print(
    """
File: src/atomate2/siesta/sets/tiers/presets/surface.py

Add this code:
"""
)

catalysis_code = """
CATALYSIS_VDW = {
    "tier": "intermediate",
    "description": (
        "Accurate catalysis with van der Waals corrections. "
        "For organic adsorbates on metal surfaces."
    ),
    "categories": ["catalysis", "surface"],
    "params": {
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [6, 6, 1],
        "Mesh.Cutoff": "300 Ry",
        "vdw": "DRSLL",                 # vdW-DF functional
        "OccupationFunction": "MP",
        "ElectronicTemperature": "300 K",
        "SCF.Mixer.Method": "Pulay",
        "SCF.Mixer.Weight": 0.005,
        "SCF.DM.Tolerance": 1e-5,
        "MD.MaxForceTol": "0.02 eV/Ang",
    },
}
"""
print(catalysis_code)

# ==============================================================================
# EXAMPLE 3: High-Throughput Screening
# ==============================================================================

print("\n" + "=" * 50)
print("Example 3: Ultra-Fast Screening")
print("=" * 50)

print(
    """
File: src/atomate2/siesta/sets/tiers/presets/performance.py

Add this code:
"""
)

htp_code = """
HIGH_THROUGHPUT = {
    "tier": "basic_dirty",
    "description": (
        "Ultra-fast for high-throughput screening. "
        "Minimal basis, coarse grids. Trends only!"
    ),
    "categories": ["performance", "structural"],
    "params": {
        "PAO.BasisSize": "SZ",          # Single-zeta
        "Mesh.Cutoff": "100 Ry",        # Very coarse
        "a2s_kpts": [2, 2, 2],
        "SCF.Mixer.Weight": 0.1,        # Fast mixing
        "SCF.DM.Tolerance": 1e-3,       # Loose convergence
        "MD.MaxForceTol": "0.1 eV/Ang",
        "Diag.Algorithm": "ELPA",
    },
}
"""
print(htp_code)

# ==============================================================================
# AUTOMATIC CLI DETECTION
# ==============================================================================

print("\n" + "=" * 50)
print("Naming Conventions for Auto-Detection")
print("=" * 50)

print(
    """
Follow these patterns for automatic categorization:

Pattern         → Category      → Example
──────────────────────────────────────────────
2d_*            → 2d            → 2d_metal
surface_*       → surface       → surface_oxide
adsorbate_*     → surface       → adsorbate_screening
magnetic_*      → magnetic      → magnetic_afm
phonon_*        → phonon        → phonon_accurate
optical_*       → optical       → optical_standard
catalysis_*     → catalysis     → catalysis_vdw (new!)
elastic_*       → mechanical    → elastic_accurate (new!)

Key rules:
- Use lowercase with underscores
- Follow existing patterns
- No manual CLI updates needed
"""
)

# ==============================================================================
# TESTING YOUR PRESET
# ==============================================================================

print("\n" + "=" * 50)
print("How to Test Your Preset")
print("=" * 50)

print(
    """
Step 1: Verify CLI detection
────────────────────────────
$ atomate2siesta-presets list
  (should show your preset)

$ atomate2siesta-presets show your_preset_name
  (should display all parameters)

Step 2: Test in Python with dry_run
────────────────────────────────────
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

maker = RelaxMaker.fixed_cell_relaxation()
maker = apply_tier_preset(maker, "your_preset_name")

job = maker.make(structure, dry_run=True)
results = run_locally(job, create_folders=True)

# Check generated siesta.fdf file

Step 3: Run test calculation
─────────────────────────────
# Remove dry_run=True for actual calculation
job = maker.make(small_structure)
results = run_locally(job, create_folders=True)
"""
)

# ==============================================================================
# COMPLETE WORKFLOW
# ==============================================================================

print("\n" + "=" * 50)
print("Complete Workflow Summary")
print("=" * 50)

print(
    """
1. Create preset in presets/<category>.py:
   MY_PRESET = {
       "tier": "intermediate",
       "description": "...",
       "categories": ["cat1", "cat2"],
       "params": {...},
   }

2. Update __init__.py:
   - Import: from .category import MY_PRESET
   - Register: "my_preset": MY_PRESET

3. Test CLI:
   $ atomate2siesta-presets list | grep my_preset
   $ atomate2siesta-presets show my_preset

4. Test in Python with dry_run=True

5. Run actual test calculation

That's it! No CLI code changes needed.
"""
)

# ==============================================================================
# BEST PRACTICES
# ==============================================================================

print("\n" + "=" * 50)
print("Best Practices")
print("=" * 50)

print(
    """
1. Start from similar existing preset
2. Choose appropriate tier level:
   - basic_dirty: Testing only
   - basic: Fast screening
   - intermediate: Standard work
   - advanced: Converged results
   - expert: Publication quality

3. Write clear descriptions:
   - Material type
   - Accuracy level
   - When to use it
   - Limitations

4. Follow naming conventions
5. Test thoroughly with known systems
6. Document parameter choices
7. Consider spin/vdW/constraints compatibility
"""
)

print("\n✅ Tutorial complete!")
print("\nSee also:")
print("  - docs/source/tier-system.rst (full documentation)")
print("  - docs/source/cli-tools.rst (CLI reference)")
