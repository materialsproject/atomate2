#!/usr/bin/env python
"""Example 1: Customizing Recipe Book with user_params.

Pass FDF parameters directly to any Recipe Book workflow.
"""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create silicon structure
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

print("\n" + "=" * 70)
print("Recipe Book with Custom Parameters")
print("=" * 70 + "\n")

# Recipe workflow with custom FDF parameters
flow = RecipeBook.eos_workflow(
    silicon,
    number_of_frames=7,  # Workflow parameter
    auto_params=False,  # Disable automatic parameter detection
    user_params={  # FDF parameters
        "PAO.BasisSize": "TZP",
        "Mesh.Cutoff": "400 Ry",
        "a2s_kpts": [6, 6, 6],
        "SCF.DM.Tolerance": 1e-6,
    },
    dry_run=True,
)

# Run it
results = run_locally(flow, create_folders=True)

print("\n✅ EOS workflow created with custom parameters!")
print("   Check siesta.fdf files in job folders\n")

print("Key points:")
print("  • user_params = your FDF parameter overrides")
print("  • auto_params=False = disable automatic parameter detection")
print("  • Works with ALL 39 Recipe Book workflows")
print("  • number_of_frames = workflow-specific parameter")
print()
