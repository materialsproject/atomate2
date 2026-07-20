#!/usr/bin/env python
"""Band structure workflow using tier presets.

Tier presets provide optimized parameters for different accuracy levels:
- basic_dirty: Quick tests (~5 min)
- basic: Standard calculations (~20 min)
- intermediate: Better accuracy (~40 min)
- advanced: High accuracy (~1-2 hours)
- expert: Publication quality (~2-4 hours)

This tutorial shows how to apply tier presets to the makers.

Runtime: Depends on tier level
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.bands import BandStructureFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker, BandStructureMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

# Load silicon primitive cell
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# ============================================================================
# CREATE MAKERS AND APPLY TIER PRESETS
# ============================================================================

# Create base makers
relax_maker = RelaxMaker.variable_cell_relaxation(use_custodian=True)
scf_maker = StaticMaker.scf(use_custodian=True)
bands_maker = BandStructureMaker.bandstructure_calculation(use_custodian=True)

# Apply tier preset to all makers
# Options (presets): relax_dirty, relax_standard, band_structure, relax_high_accuracy
TIER = "band_structure"

relax_maker = apply_tier_preset(relax_maker, TIER)
scf_maker = apply_tier_preset(scf_maker, TIER)
bands_maker = apply_tier_preset(bands_maker, TIER)

# ============================================================================
# CREATE WORKFLOW WITH PRESET MAKERS
# ============================================================================

maker = BandStructureFlowMaker(
    relax_maker=relax_maker,
    scf_maker=scf_maker,
    bands_maker=bands_maker,
    plot_bands=True,
)

# Generate and run workflow
flow = maker.make(structure)
results = run_locally(flow, create_folders=True, root_dir="04_with_tier_preset")

print("\n" + "=" * 60)
print(f"Band Structure with '{TIER}' Tier Complete!")
print("=" * 60)
print(f"\nUsed tier preset: {TIER}")
print("Tier presets automatically configure:")
print("  - Basis set size")
print("  - Mesh cutoff")
print("  - K-point density")
print("  - Convergence tolerances")
