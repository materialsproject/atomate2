"""Interlayer Distance Optimization for Heterostructures

Demonstrates automatic optimization of interlayer spacing to find
the optimal van der Waals separation between 2D layers.
"""

from pymatgen.core import Lattice, Structure

from atomate2.siesta.flows.heterostructures import InterfaceFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

# Create graphene structure using pymatgen (for bilayer)
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
# Production mode: Distance scanning with custodian
relax_maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,  # Recommended for multiple relaxations
    custodian_max_errors=10,
)
relax_maker = apply_tier_preset(
    relax_maker,
    "2d_vdw",  # Includes DFTD3 van der Waals corrections automatically
)

# Dry-run mode to preview distance scan:
# relax_maker = RelaxMaker.fixed_cell_relaxation(dry_run=True, dry_run_format="full")
# relax_maker = apply_tier_preset(relax_maker, "2d_vdw")

# InterfaceFlowMaker with distance optimization
interface_maker = InterfaceFlowMaker(
    name="Graphene_Bilayer_Distance_Opt",
    relax_maker=relax_maker,
    interlayer_distance=3.5,  # Initial guess
    matching_mode="strain",  # Same material, so strain mode is fine
    optimize_interlayer_distance=True,
    distance_range=(3.0, 4.0),  # Scan from 3.0 to 4.0 Angstrom
    distance_steps=11,  # 11 points = 0.1 A spacing
    calculate_binding_energy=True,
    dry_run=True,
)

# Generate workflow
flow = interface_maker.make(
    bottom_layer=graphene,
    top_layer=graphene,  # Bilayer
)

# Run
results = run_locally(flow, create_folders=True)

print("\n" + "=" * 70)
print("INTERLAYER DISTANCE OPTIMIZATION")
print("=" * 70)
print("Scans interlayer distances to find optimal van der Waals separation.")
print("\nOutputs include:")
print("  - distance_vs_energy.png (binding energy curve)")
print("  - optimal_distance.txt")
print("  - interface_optimized.cif")
print("\nNOTE: If dry_run=True was used, these are NOT calculated values!")
print("      Dry-run only generates input files for validation.")
