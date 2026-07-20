#!/usr/bin/env python
"""Tutorial: Restarting from Previous Geometry (XV File)

Two-stage relaxation workflow demonstrating XV restart:
1. Quick relaxation (loose tolerance 0.04 eV/Å) → saves XV
2. Fine relaxation (tight tolerance 0.01 eV/Å) → reads XV, needs only ~5-10 steps

See README.md for detailed physics background and analysis tips.
"""

from pymatgen.core import Structure
from jobflow import Flow, run_locally
from atomate2.siesta.jobs.core import RelaxMaker

# Configuration
STRUCTURE_FILE = "../../00-structures/Si_mp-149_conventional_standard.cif"
OUTPUT_DIR = "xv_restart_workflow"

# Read structure
structure = Structure.from_file(STRUCTURE_FILE)
print(f"Loaded: {structure.composition.reduced_formula} ({structure.num_sites} atoms)")

# Step 1: Quick relaxation (saves XV)
quick_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "a2s_kpts": [4, 4, 4],
        "PAO.BasisSize": "DZ",
        "Mesh.Cutoff": "200 Ry",
        "MD.MaxForceTol": "0.04 eV/Ang",  # Loose tolerance
        "MD.MaxCGDispl": "0.2 Bohr",
        "MD.NumCGsteps": 50,
        "MD.UseSaveXV": True,  # Save XV for reuse
    },
)
quick_maker.name = "Quick Relax (Generate XV)"
quick_job = quick_maker.make(structure)

# Step 2: Fine relaxation (reads XV from Step 1)
fine_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "a2s_kpts": [4, 4, 4],
        "PAO.BasisSize": "DZ",
        "Mesh.Cutoff": "200 Ry",
        "MD.MaxForceTol": "0.01 eV/Ang",  # Tight tolerance
        "MD.MaxCGDispl": "0.1 Bohr",
        "MD.NumCGsteps": 100,
        "MD.UseSaveXV": True,  # Read previous XV
    },
    copy_siesta_kwargs={
        "restart_to_input": True,  # Copy DM file
        "additional_siesta_files": ["siesta.XV"],  # Also copy XV file
    },
)
fine_maker.name = "Fine Relax (XV Restart)"
fine_job = fine_maker.make(structure, prev_dir=quick_job.output.dir_name)

# Create workflow
workflow = Flow([quick_job, fine_job], name="XV Restart Workflow")
print("Created 2-step workflow: Quick → Fine (XV restart)")

# Run in dry-run mode (previews workflow, no actual calculation)
print("\nPreviewing workflow structure...\n")
responses = run_locally(workflow, create_folders=True, root_dir=OUTPUT_DIR)

print(f"\n✓ Dry-run complete! Check: {OUTPUT_DIR}/")
print("  Verify: grep 'MD.UseSaveXV' */siesta.fdf")
print("  Compare: grep 'MD.MaxForceTol' */siesta.fdf\n")
