"""MoS2/Graphene Heterostructure with Supercell Matching

Demonstrates automatic supercell generation for heterostructures
with different lattice constants. Uses supercell mode instead of strain.
"""

from pymatgen.core import Lattice, Structure

from atomate2.siesta.flows.heterostructures import InterfaceFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

# Create MoS2 monolayer structure using pymatgen
mos2_lattice = Lattice.from_parameters(
    a=3.16, b=3.16, c=20.0, alpha=90, beta=90, gamma=120
)
mos2 = Structure(
    lattice=mos2_lattice,
    species=["Mo", "S", "S"],
    coords=[[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.58], [2 / 3, 1 / 3, 0.42]],
)

# Create graphene structure using pymatgen
graphene_lattice = Lattice.from_parameters(
    a=2.46, b=2.46, c=20.0, alpha=90, beta=90, gamma=120
)
graphene = Structure(
    lattice=graphene_lattice,
    species=["C", "C"],
    coords=[[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.5]],
)

# Lattice mismatch: (3.16 - 2.46) / 2.46 = 28% → Need supercell mode!

# ============================================================================
# EXECUTION MODE OPTIONS
# ============================================================================
# Production mode with custodian (recommended for large supercells!)
relax_maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,  # Essential for large supercells (SCF convergence issues)
    custodian_max_errors=10,
)
relax_maker = apply_tier_preset(relax_maker, "2d_vdw")

# Dry-run mode to preview supercell size (NO execution):
# relax_maker = RelaxMaker.fixed_cell_relaxation(dry_run=True, dry_run_format="full")
# relax_maker = apply_tier_preset(relax_maker, "2d_vdw")

# InterfaceFlowMaker with supercell mode
interface_maker = InterfaceFlowMaker(
    name="MoS2_Graphene_Supercell",
    relax_maker=relax_maker,
    interlayer_distance=3.5,
    matching_mode="supercell",  # Use supercell matching (28% mismatch!)
    max_supercell_size=10,  # Maximum supercell dimension (increased for large mismatch)
    max_area_mismatch=0.10,  # 10% maximum tolerance (increased for 28% initial mismatch)
    optimize_interlayer_distance=True,  # Disable to speed up for tutorial
    calculate_binding_energy=True,  # Disable to speed up for tutorial
    dry_run=True,
)

# Generate workflow
flow = interface_maker.make(
    bottom_layer=mos2,
    top_layer=graphene,
)

# Run
results = run_locally(flow, create_folders=True)

print("\n" + "=" * 70)
print("MoS2/GRAPHENE SUPERCELL INTERFACE")
print("=" * 70)
print("Supercell matching creates minimal strain by finding")
print("commensurate supercells for both layers.")
print("\nCheck outputs for supercell dimensions and strain analysis.")
print("\nNOTE: If dry_run=True was used, these are NOT calculated values!")
print("      Dry-run only generates input files for validation.")
