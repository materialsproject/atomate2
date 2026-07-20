#!/usr/bin/env python
"""
EOS with Custom Mesh.Cutoff via Powerups.

This tutorial demonstrates using the update_user_siesta_settings powerup
to modify Mesh.Cutoff and other parameters after creating the workflow.

Key concepts:
- Powerups: Modify job parameters after workflow creation
- More flexible than user_params
- Can apply different settings to different jobs
- Useful for fine-tuning existing workflows

Powerup functions:
- update_user_siesta_settings: Update SIESTA parameters
- update_siesta_custodian_handlers: Modify error handlers
- set_dry_run: Switch between dry-run and real calculations
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.eos import SiestaEosFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings

# Load structure
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")


# Method 3: Add parameters not in user_params
# ============================================
print("\nMethod 3: Add new parameters via powerup")

# Create basic maker
maker3 = SiestaEosFlowMaker(
    dry_run=True,
)

workflow3 = maker3.make(structure)

# Add basis, mesh cutoff and other parameters via powerup
workflow3 = update_user_siesta_settings(
    workflow3,
    {
        "PAO.BasisSize": "DZP",  # Add new parameter
        "Mesh.cutoff": "400 Ry",  # Add new parameter
        "a2s_kpts": [6, 6, 6],  # Add new parameter
        # "XC.functional": "GGA",  # Add new parameter
        # "XC.authors": "PBE",  # Add new parameter
        "SCF.Mixer.Weight": 0.05,  # Fine-tune SCF convergence
    },
)

results3 = run_locally(workflow3, create_folders=True)

print("✓ Added parameters via powerup complete!")
print("  - Mesh.Cutoff: 400 Ry (added)")
print("  - XC functional: GGA-PBE (added)")
print("  - SCF mixer weight: 0.05 (added)")
