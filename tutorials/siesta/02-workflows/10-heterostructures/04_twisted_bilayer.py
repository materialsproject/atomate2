"""Twisted Bilayer Graphene (Magic Angle)

Demonstrates twisted bilayer heterostructures with rotation.
This example creates a twisted bilayer graphene structure at the
first magic angle (θ ≈ 1.1°) using automatic supercell generation.
"""

from pymatgen.core import Lattice, Structure

from atomate2.siesta.flows.heterostructures import InterfaceFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

# Create graphene structure using pymatgen
graphene_lattice = Lattice.from_parameters(
    a=2.46, b=2.46, c=20.0, alpha=90, beta=90, gamma=120
)
graphene = Structure(
    lattice=graphene_lattice,
    species=["C", "C"],
    coords=[[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.5]],
)

# ============================================================================
# EXECUTION MODE OPTIONS
# ============================================================================
# Production mode: Twisted bilayer with custodian
relax_maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,  # Essential for twisted bilayers (large supercells!)
    custodian_max_errors=10,
)
relax_maker = apply_tier_preset(
    relax_maker,
    "2d_vdw",  # Includes DFTD3 van der Waals corrections automatically
)

# Dry-run mode to check supercell size before calculation:
# relax_maker = RelaxMaker.fixed_cell_relaxation(dry_run=True, dry_run_format="full")
# relax_maker = apply_tier_preset(relax_maker, "2d_vdw")

# InterfaceFlowMaker with twist angle
# Note: Small twist angles (like magic angles) create large supercells!
# For production, use θ ≈ 1.1° (first magic angle)
# For testing, use larger angles like 21.8° or 13.2°
interface_maker = InterfaceFlowMaker(
    name="Twisted_Bilayer_Graphene",
    relax_maker=relax_maker,
    interlayer_distance=3.35,  # Typical graphene interlayer spacing
    matching_mode="supercell",  # Use supercell matching for twisted structures
    max_supercell_size=10,  # Larger limit for twisted structures
    optimize_interlayer_distance=True,
    calculate_binding_energy=True,
    dry_run=True,
)

# Generate workflow with rotation angle
flow = interface_maker.make(
    bottom_layer=graphene,
    top_layer=graphene,  # Same material (twisted bilayer)
    rotation_angle=25,  # 21.8,  # Testing angle (use ~1.1° for magic angle)
)

# Run
results = run_locally(flow, create_folders=True)

print("\n" + "=" * 70)
print("TWISTED BILAYER GRAPHENE")
print("=" * 70)
print("Twist angle: 21.8° (testing)")
print("For magic angle physics, use θ ≈ 1.1° (requires larger supercell)")
print("\nMagic angles in twisted bilayer graphene:")
print("  - First magic angle: θ ≈ 1.1° (flat bands, correlated phases)")
print("  - Second magic angle: θ ≈ 0.5°")
print("\nNote: Small twist angles create very large moiré supercells!")
print("      Use testing angles (13-22°) for quick validation.")
print("\nNOTE: If dry_run=True was used, these are NOT calculated values!")
print("      Dry-run only generates input files for validation.")
