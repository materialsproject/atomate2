#!/usr/bin/env python
"""
Full Basis Parameter Optimization with Powerups.

This tutorial demonstrates using powerups to customize EOSFullBasisConvergenceFlowMaker
workflows after they are created.

Key concepts:
- Powerups: Modify job parameters after workflow creation
- update_user_siesta_settings: Update SIESTA parameters for all jobs
- Useful for adding Mesh.Cutoff, SCF settings, spin polarization
- Can fine-tune workflows without modifying the maker

Common powerup functions:
- update_user_siesta_settings: Update SIESTA FDF parameters
- update_siesta_custodian_handlers: Modify error handlers
- set_dry_run: Switch between dry-run and real calculations

When to use powerups with EOSFullBasisConvergenceFlowMaker:
- Adding parameters not available in the maker constructor
- Fine-tuning SCF convergence settings
- Adding spin polarization or other physics settings
- Modifying mesh cutoff for all calculations uniformly
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.eos import EOSFullBasisConvergenceFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings

# Load structure
# structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")
structure = Structure.from_file("../../../00-structures/MoS2.cif")

# =============================================================================
# Method: Create workflow and customize with powerups
# =============================================================================
print("Full Basis Convergence with Powerups")
print("=" * 50)

# Step 1: Create the basic workflow with custodian
maker = EOSFullBasisConvergenceFlowMaker(
    # dry_run=True,  # Use dry_run for testing
    basis_sizes=["DZ", "DZP"],
    energy_shifts=[0.01, 0.015],
    split_norms=[0.15, 0.20],
    linear_strain=(-0.04, 0.04),
    number_of_frames=5,
    a2s_kpts=[2, 2, 2],
    # Custodian error handling
    use_custodian=True,
    custodian_max_errors=10,
)

workflow = maker.make(structure)

# Step 2: Apply powerups to customize all jobs in the workflow
# This will update ALL jobs (all basis/ES/SN combinations, all strain points)
workflow = update_user_siesta_settings(
    workflow,
    {
        # Mesh cutoff for higher accuracy
        "Mesh.Cutoff": "200 Ry",
        # SCF convergence settings (helpful for difficult parameter combinations)
        "SCF.Mixer.Weight": 0.05,
        "SCF.DM.Tolerance": 1.0e-5,
        "MaxSCFIterations": 300,
        # Electronic temperature for better convergence
        "ElectronicTemperature": "300 K",
    },
)

# Step 3: Run the workflow
results = run_locally(workflow, create_folders=True, root_dir="03_powerups")

print("\n✓ Full basis convergence with powerups complete!")
print("  - Basis sizes: DZ, DZP")
print("  - Energy shifts: 0.01, 0.015 Ry")
print("  - Split norms: 0.15, 0.20")
print("  - Total combinations: 2 × 2 × 2 = 8 EOS calculations")
print("  - Mesh.Cutoff: 350 Ry (applied via powerup)")
print("  - SCF settings: fine-tuned for convergence")
print("  - Custodian: enabled for automatic error recovery")
