"""Basic Heterostructure: Graphene/h-BN Interface

Demonstrates InterfaceFlowMaker for creating 2D heterostructures.
This example creates a graphene/h-BN interface with lattice matching.
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

# Create h-BN structure using pymatgen
hBN_lattice = Lattice.from_parameters(
    a=2.50, b=2.50, c=20.0, alpha=90, beta=90, gamma=120
)
hBN = Structure(
    lattice=hBN_lattice,
    species=["B", "N"],
    coords=[[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.5]],
)

# Lattice mismatch: (2.50 - 2.46) / 2.46 = 1.6% → Good for strain mode!

# ============================================================================
# EXECUTION MODE OPTIONS
# ============================================================================
# Choose one of the following execution modes:

# 1. PRODUCTION MODE (default): Full calculation with automatic error handling
relax_maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,  # Enable automatic error recovery (RECOMMENDED!)
    custodian_max_errors=10,  # Retry up to 10 times
)
relax_maker = apply_tier_preset(relax_maker, "2d_vdw")

# 2. DRY-RUN MODE: Generate input files only (NO execution, 99.9% faster!)
# Uncomment to test workflow structure without running calculations:
# relax_maker = RelaxMaker.fixed_cell_relaxation(
#     dry_run=True,
#     dry_run_format="full",  # Generate all input files
# )
# relax_maker = apply_tier_preset(relax_maker, "2d_vdw")

# 3. FAST TESTING: Use dirty preset for quick validation (10-20x faster)
# WARNING: Results NOT quantitatively accurate!
# relax_maker = RelaxMaker.fixed_cell_relaxation(use_custodian=True)
# relax_maker = apply_tier_preset(relax_maker, "2d_vdw_dirty")

# Create InterfaceFlowMaker with strain matching
interface_maker = InterfaceFlowMaker(
    name="Graphene_hBN_Interface",
    relax_maker=relax_maker,
    interlayer_distance=3.4,  # Initial separation (Angstrom)
    matching_mode="strain",  # Simple strain matching (1.6% mismatch)
    apply_strain_to="smaller",  # Strain graphene (smaller lattice)
    optimize_interlayer_distance=True,
    calculate_binding_energy=True,
    dry_run=True,
)

# Generate workflow
flow = interface_maker.make(
    bottom_layer=graphene,
    top_layer=hBN,
)

# Run locally (or submit to remote)
results = run_locally(flow, create_folders=True)

print("\n" + "=" * 70)
print("GRAPHENE/h-BN INTERFACE RESULTS")
print("=" * 70)
print(f"Interface created: {results[flow.output.uuid][1].output}")
print("\nCheck output directory for:")
print("  - interface_relaxed.cif")
print("  - interface_analysis.json")
print("  - binding_energy.txt")
print("\nNOTE: If dry_run=True was used, these are NOT calculated values!")
print("      Dry-run only generates input files for validation.")
