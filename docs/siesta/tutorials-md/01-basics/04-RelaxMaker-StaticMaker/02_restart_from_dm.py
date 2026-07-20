#!/usr/bin/env python
"""Tutorial: Restarting from Previous Density Matrix (DM)

Two-step workflow demonstrating DM restart for faster SCF convergence:
1. Coarse calculation (2x2x2 k-points) → saves DM
2. Fine calculation (4x4x4 k-points) → reads DM, converges in ~3-5 SCF iterations

See README.md for detailed physics background and analysis tips.
"""

from pymatgen.core import Structure
from jobflow import Flow, run_locally
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker

# Configuration
STRUCTURE_FILE = "../../00-structures/Si_mp-149_primitive.cif"
OUTPUT_DIR = "dm_restart_workflow"

# Read structure
structure = Structure.from_file(STRUCTURE_FILE)
print(f"Loaded: {structure.composition.reduced_formula} ({structure.num_sites} atoms)")

# Step 1: Coarse relaxation (saves DM)
coarse_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "a2s_kpts": [2, 2, 2],
        "PAO.BasisSize": "DZ",
        "DM.Tolerance": 1e-4,
        "DM.UseSaveDM": True,  # Save DM for reuse
        "Mesh.Cutoff": "200 Ry",
    },
)
coarse_maker.name = "Coarse Relax (Generate DM)"
coarse_job = coarse_maker.make(structure)

# Step 2: Fine calculation (reads DM from Step 1)
fine_maker = StaticMaker.scf(
    user_params={
        "a2s_kpts": [4, 4, 4],  # Refined k-mesh
        "PAO.BasisSize": "DZ",  # MUST match Step 1
        "DM.Tolerance": 1e-5,  # Tighter convergence
        "DM.UseSaveDM": True,  # Read previous DM
        "Mesh.Cutoff": "250 Ry",
    },
    copy_siesta_kwargs={"restart_to_input": True},  # Copy DM file from prev_dir
)
fine_maker.name = "Fine Calculation (DM Restart)"
fine_job = fine_maker.make(structure, prev_dir=coarse_job.output.dir_name)

# Create workflow
workflow = Flow([coarse_job, fine_job], name="DM Restart Workflow")
print("Created 2-step workflow: Coarse → Fine (DM restart)")

# Run in dry-run mode (previews workflow, no actual calculation)
print("\nPreviewing workflow structure...\n")
responses = run_locally(workflow, create_folders=True, root_dir=OUTPUT_DIR)

print(f"\n✓ Dry-run complete! Check: {OUTPUT_DIR}/")
print("  Verify: grep 'DM.UseSaveDM' */siesta.fdf")
print("  Compare: grep 'kgrid' */siesta.fdf\n")
