"""Compare ALL available finite-size correction schemes."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects import (
    CorrectionComparisonFlowMaker,
    create_vacancy_with_ghost,
)
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

# Create MgO 2×2×2 supercell
lattice = Lattice.cubic(4.212)
unit_cell = Structure(
    lattice,
    ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"],
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ],
)
host_structure = unit_cell.make_supercell([2, 2, 2])

# Create charged vacancy
o_indices = [i for i, site in enumerate(host_structure) if site.specie.symbol == "O"]
defect_structure = create_vacancy_with_ghost(host_structure, o_indices[15])

print("Comparing ALL available correction schemes...")

# Create makers with tier presets
defect_relax_maker = apply_tier_preset(
    RelaxMaker.fixed_cell_relaxation(), "defect_dirty"
)
host_static_maker = apply_tier_preset(StaticMaker(), "defect_dirty")

# AUTO-ENABLED file writing:
# - .VT files: freysoldt, kumagai, slab2d
# - .RHO files: makov-payne-quadrupole

# Create comparison workflow with ALL 6 correction schemes
maker = CorrectionComparisonFlowMaker(
    epsilon_static=9.8,  # Fallback for isotropic schemes
    defect_type="vacancy",
    charge_state=2,
    correction_schemes=[
        "lany-zunger",  # 1. Simple (monopole, no files)
        "makov-payne",  # 2. Monopole only (Q=0)
        "makov-payne-quadrupole",  # 3. Monopole + quadrupole from .RHO
        "freysoldt",  # 4. Potential alignment from .VT
        "kumagai",  # 5. SOTA with atomic-site sampling
        "slab2d",  # 6. For 2D materials (needs ε∥, ε⊥)
    ],
    epsilon_parallel=6.8,  # In-plane (ε∥) for Slab2D
    epsilon_perpendicular=3.0,  # Out-of-plane (ε⊥) for Slab2D
    auto_calculate_chemical_potentials=True,
    dry_run=False,
    skip_relax=True,
    defect_relax_maker=defect_relax_maker,
    host_static_maker=host_static_maker,
)

flow = maker.make(
    defect_structure,
    host_structure,
    host_structure[o_indices[15]].frac_coords.tolist(),
    "O",
)

# Run and get comparison results
results = run_locally(flow, create_folders=True, ensure_success=True)

# Access summary job output (find by name)
summary_job = [job for job in flow.jobs if job.name.endswith("_summary")][0]
summary = results[summary_job.uuid][1].output

print(f"\n{summary['summary_text']}")
print(f"\nRecommendation: {summary['recommendation']}")

# Scheme comparison
print("\n" + "=" * 80)
print("CORRECTION SCHEME COMPARISON")
print("=" * 80)
print("\n1. Lany-Zunger:")
print("   • Simple monopole (Q=0)")
print("   • Fast, no extra files")
print("   • Good for initial estimates")
print("\n2. Makov-Payne (basic):")
print("   • Monopole only (Q=0)")
print("   • No .RHO files needed")
print("   • Conservative baseline")
print("\n3. Makov-Payne-Quadrupole:")
print("   • Monopole + quadrupole from .RHO")
print("   • Best for anisotropic defects")
print("   • See how much Q matters!")
print("\n4. Freysoldt:")
print("   • Potential alignment from .VT")
print("   • Gold standard for isotropic systems")
print("   • Widely used in literature")
print("\n5. Kumagai (SOTA):")
print("   • Atomic-site sampling from .VT")
print("   • Best for relaxed systems")
print("   • Publication-quality accuracy")
print("\n6. Slab2D (not shown here):")
print("   • For 2D materials with vacuum")
print("   • Requires ε∥ and ε⊥")
print("   • See Tutorial 08 for example")
print("=" * 80)
print("\n✓ This comparison quantifies uncertainty from correction scheme choice!")
print("✓ Spread > 0.1 eV suggests larger supercell needed")
