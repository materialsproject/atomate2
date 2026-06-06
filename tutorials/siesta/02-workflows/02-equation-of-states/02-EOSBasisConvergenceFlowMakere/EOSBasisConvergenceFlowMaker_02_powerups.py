#!/usr/bin/env python
"""
EOS Basis Convergence with Powerups.

This tutorial demonstrates using powerups to customize EOS basis convergence
workflows after they are created.

Key concepts:
- Powerups: Modify job parameters after workflow creation
- update_user_siesta_settings: Update SIESTA parameters for all jobs
- Useful for adding parameters like Mesh.Cutoff, k-points, SCF settings
- Can fine-tune workflows without modifying the maker

Common powerup functions:
- update_user_siesta_settings: Update SIESTA FDF parameters
- update_siesta_custodian_handlers: Modify error handlers
- set_dry_run: Switch between dry-run and real calculations
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.basis import EOSBasisConvergenceFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings

# Load structure
structure = Structure.from_file("../../../00-structures/MoS2.cif")

# =============================================================================
# Method: Create workflow and customize with powerups
# =============================================================================
print("EOS Basis Convergence with Powerups")
print("=" * 50)

# Step 1: Create the basic workflow
maker = EOSBasisConvergenceFlowMaker(
    basis_sets=["SZ", "DZ", "DZP", "DZDP"],
    linear_strain=(-0.05, 0.05),
    number_of_frames=7,
    # dry_run=True,  # Use dry_run for testing
    # dry_run_output_dir="02_powerups",
)

workflow = maker.make(structure)

# Step 2: Apply powerups to customize all jobs in the workflow
# This will update ALL jobs (all basis sets, all strain points)
workflow = update_user_siesta_settings(
    workflow,
    {
        # Mesh cutoff for accuracy
        "Mesh.Cutoff": "200 Ry",
        # K-points for 2D material (MoS2)
        "a2s_kpts": [2, 2, 1],
        # SCF convergence settings
        "SCF.Mixer.Weight": 0.05,
        "SCF.DM.Tolerance": 1.0e-5,
        # Electronic temperature for metals/semiconductors
        "ElectronicTemperature": "300 K",
    },
)

# Step 3: Run the workflow
results = run_locally(workflow, create_folders=True, root_dir="02_powerups")

print("\n✓ EOS basis convergence with powerups complete!")
