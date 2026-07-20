"""Automatic Mode Selection for Heterostructures

Demonstrates the 'auto' mode which intelligently selects between
strain matching and supercell matching based on lattice mismatch.

Rules:
- Small mismatch (<5%): Use strain mode (simpler, smaller cells)
- Large mismatch (>5%): Use supercell mode (minimal strain)
"""

from pymatgen.core import Lattice, Structure

from atomate2.siesta.flows.heterostructures import InterfaceFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

# Example 1: Small mismatch - Graphene/h-BN (~1.6% mismatch)
# Should automatically select "strain" mode

# Create graphene structure using pymatgen
graphene_lattice = Lattice.from_parameters(
    a=2.46, b=2.46, c=20.0, alpha=90, beta=90, gamma=120
)
graphene = Structure(
    lattice=graphene_lattice,
    species=["C", "C"],
    coords=[[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.5]],
)

# Create h-BN structure using pymatgen
hBN_lattice = Lattice.from_parameters(
    a=2.50, b=2.50, c=20.0, alpha=90, beta=90, gamma=120
)
hBN = Structure(
    lattice=hBN_lattice,
    species=["B", "N"],
    coords=[[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.5]],
)

# ============================================================================
# EXECUTION MODE OPTIONS
# ============================================================================
# Production mode with custodian (auto mode may create large supercells!)
relax_maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,
    custodian_max_errors=10,
)
relax_maker = apply_tier_preset(relax_maker, "2d_vdw")

# Dry-run mode to see which mode (strain/supercell) will be selected:
# relax_maker = RelaxMaker.fixed_cell_relaxation(dry_run=True, dry_run_format="full")
# relax_maker = apply_tier_preset(relax_maker, "2d_vdw")

# InterfaceFlowMaker with AUTO mode
interface_maker = InterfaceFlowMaker(
    name="Auto_Mode_Small_Mismatch",
    relax_maker=relax_maker,
    interlayer_distance=3.4,
    matching_mode="auto",  # Automatically select best mode
    max_lattice_mismatch=0.05,  # 5% threshold (default)
    optimize_interlayer_distance=True,
    calculate_binding_energy=True,
)

# Generate workflow for small mismatch case
flow_small = interface_maker.make(
    bottom_layer=graphene,
    top_layer=hBN,
)

print("\n" + "=" * 70)
print("AUTO MODE SELECTION - SMALL MISMATCH")
print("=" * 70)
print("System: Graphene/h-BN (1.6% lattice mismatch)")
print("Expected mode: STRAIN (mismatch < 5%)")
print("Result: One layer strained to match the other")

# Example 2: Large mismatch - MoS2/graphene (~28% mismatch)
# Should automatically select "supercell" mode

# Create MoS2 structure using pymatgen
mos2_lattice = Lattice.from_parameters(
    a=3.16, b=3.16, c=20.0, alpha=90, beta=90, gamma=120
)
mos2 = Structure(
    lattice=mos2_lattice,
    species=["Mo", "S", "S"],
    coords=[[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.58], [2 / 3, 1 / 3, 0.42]],
)

# InterfaceFlowMaker with AUTO mode for large mismatch
interface_maker_large = InterfaceFlowMaker(
    name="Auto_Mode_Large_Mismatch",
    relax_maker=relax_maker,
    interlayer_distance=3.5,
    matching_mode="auto",  # Automatically select best mode
    max_supercell_size=5,  # For supercell mode
    max_lattice_mismatch=0.05,  # 5% threshold
    optimize_interlayer_distance=True,
    calculate_binding_energy=True,
)

# Generate workflow for large mismatch case
flow_large = interface_maker_large.make(
    bottom_layer=mos2,
    top_layer=graphene,
)

print("\n" + "=" * 70)
print("AUTO MODE SELECTION - LARGE MISMATCH")
print("=" * 70)
print("System: MoS₂/graphene (28% lattice mismatch)")
print("Expected mode: SUPERCELL (mismatch > 5%)")
print("Result: Commensurate supercells with minimal strain")
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("Use 'auto' mode unless you have specific requirements:")
print("  - Prefer strain: set matching_mode='strain'")
print("  - Prefer supercell: set matching_mode='supercell'")
print("  - Let code decide: set matching_mode='auto' (recommended)")
print("\nNOTE: If dry_run=True was used, these are NOT calculated values!")
print("      Dry-run only generates input files for validation.")

# Run both examples
# results = run_locally([flow_small, flow_large], create_folders=True)
