#!/usr/bin/env python3
"""ASE NEB with endpoint relaxation.

This tutorial demonstrates running NEB with automatic relaxation of
the initial and final structures. This ensures endpoints are at local
minima, which is required for a valid NEB calculation.

Workflow:
1. Relax initial structure (fixed cell)
2. Relax final structure (fixed cell)
3. Generate NEB images from relaxed endpoints
4. Run NEB optimization with persistent folders
5. Plot energy profile

Benefits:
✓ Ensures endpoints are local minima (required for valid NEB)
✓ Improves NEB convergence
✓ More accurate barrier heights
✓ Automatic - just set relax_endpoints=True
"""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.neb import AseNebFlowMaker
from atomate2.siesta.jobs.core import StaticMaker, RelaxMaker
from atomate2.siesta.powerups import update_user_siesta_settings

# Load initial and final structures RELAXED
initial = Structure.from_file("initial.xsf")
final = Structure.from_file("final.xsf")
# Load initial and final structures
# initial = Structure.from_file("../../../00-structures/mgo_li-initial.xsf")
# final = Structure.from_file("../../../00-structures/mgo_li-final.xsf")


print("ASE NEB with Endpoint Relaxation")
print("=" * 60)
print(f"Structure: {initial.composition}")
print("Initial → Final: Vacancy diffusion")
print()

# =============================================================================
# Create ASE NEB workflow WITH endpoint relaxation
# =============================================================================

# Configure makers
static_maker = StaticMaker()
relax_maker = RelaxMaker.fixed_cell_relaxation()  # Optional: customize relax settings

# Create NEB maker with endpoint relaxation enabled
maker = AseNebFlowMaker(
    # dry_run=True,
    number_of_images=5,
    # optimizer="FIRE",
    optimizer="PER_IMAGE_BFGS",
    fmax=0.05,
    climbing_image=True,
    spring_constant=1.0,
    relax_endpoints=False,  # ← Enable endpoint relaxation!
    relax_maker=relax_maker,  # Optional: use custom relax maker
    static_maker=static_maker,
)

print("Workflow steps:")
print("  1. Relax initial structure (ensure it's a local minimum)")
print("  2. Relax final structure (ensure it's a local minimum)")
print("  3. Generate NEB images from relaxed structures")
print("  4. Run NEB optimization (persistent folders)")
print("  5. Plot energy profile")
print()

# Create flow
flow = maker.make(initial_structure=initial, final_structure=final)

# Apply SIESTA parameters (applied to ALL jobs: relax + NEB)
flow = update_user_siesta_settings(
    flow,
    {
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [1, 1, 1],
        "Mesh.Cutoff": "50 Ry",
        "xc.functional": "GGA",
        "xc.authors": "PBE",
        "a2s_pseudo_relativistic": "SR",
    },
)

print("SIESTA parameters (applied to all calculations):")
print("  Basis: DZP")
print("  K-points: 2×2×2")
print("  Cutoff: 50 Ry")
print()

# Run the workflow
print("Starting NEB workflow with endpoint relaxation...")
print("=" * 60)
results = run_locally(
    flow, create_folders=True, root_dir="01_ase_neb_with_relax-PerImageBFGS"
)

print("\n" + "=" * 60)
print("✓ NEB workflow complete!")
print("=" * 60)
print("\nFolder structure:")
print("  job_XXX_Initial_Relaxation/      (relax initial structure)")
print("  job_YYY_Final_Relaxation/        (relax final structure)")
print("  job_ZZZ_Generate_Images/         (interpolate from relaxed)")
print("    ├── image_0.xyz/.cif")
print("    └── ...")
print("  job_AAA_NEB_Optimization/        (NEB with persistent folders)")
print("    ├── neb_progress.log           ← Track progress here!")
print("    ├── image_0/")
print("    └── ...")
print()
print("Files to check:")
print("  1. Relaxed structures:")
print("     - job_*_Initial_Relaxation/siesta.XV")
print("     - job_*_Final_Relaxation/siesta.XV")
print("  2. NEB progress: job_*_NEB_Optimization/neb_progress.log")
print("  3. Final results: job_*_NEB_Optimization/ase_neb_info.txt")
print("  4. Energy plot: neb_energy_profile.png")
print()

print("Why relax endpoints?")
print("-" * 60)
print("  ✓ NEB requires endpoints to be at local minima")
print("  ✓ If endpoints have residual forces, NEB path is invalid")
print("  ✓ Relaxation ensures accurate barrier heights")
print("  ✓ Improves NEB convergence (fewer iterations needed)")
print()

print("When NOT to relax endpoints:")
print("-" * 60)
print("  • Endpoints already fully relaxed in previous calculation")
print("  • Testing/debugging with approximate structures")
print("  • Comparing multiple NEB paths with same endpoints")
print()

print("Tip: Monitor the relaxation forces to see if relaxation was needed:")
print("  grep 'max' job_*_Relaxation/siesta.out | tail -1")
