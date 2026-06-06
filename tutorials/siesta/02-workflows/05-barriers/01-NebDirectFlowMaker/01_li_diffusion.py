#!/usr/bin/env python
"""NEB vacancy diffusion with 5 images using NebDirectMaker.

This tutorial demonstrates three methods to customize SIESTA parameters:
1. Pass neb_maker with user_params at initialization (like RelaxMaker style)
2. Configure neb_maker input_set_generator (at neb_maker level)
3. Using powerups to update flow settings (RECOMMENDED - at flow level)
"""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.neb import NebDirectFlowMaker
from atomate2.siesta.jobs.core import LuaMaker

# Load initial and final structures
initial = Structure.from_file("../../../00-structures/mgo_li-initial.xsf")
final = Structure.from_file("../../../00-structures/mgo_li-final.xsf")

# =============================================================================
# Method 1: Pass neb_maker with user_params (like RelaxMaker.fixed_cell_relaxation)
# =============================================================================
# Create a custom neb_maker with user_params, then pass to NebDirectMaker
maker = NebDirectFlowMaker(
    # dry_run=True,
    number_of_images=5,
    relax_endpoints=False,
    neb_maker=LuaMaker.neb(
        use_custodian=True,  # Enable automatic error handling
        user_params={
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [1, 1, 1],
            "Mesh.Cutoff": "50 Ry",
            "xc.functional": "GGA",
            "xc.authors": "PBE",  # Good default for solids
            "a2s_pseudo_relativistic": "SR",
        },
    ),
)

# =============================================================================
# Method 2: Configure neb_maker input_set_generator directly
# =============================================================================
# This sets parameters on the NEB calculation maker specifically
# maker.neb_maker.input_set_generator.user_params = {
#    "PAO.BasisSize": "DZP",
#    "a2s_kpts": [2, 2, 2],
#    "Mesh.Cutoff": "50 Ry",
#    "xc.functional": "GGA",
#    "xc.authors": "PBE",  # Good default for solids
# }

flow = maker.make(initial_structure=initial, final_structure=final)

# =============================================================================
# Method 3: Use powerups to update settings (RECOMMENDED)
# =============================================================================
# This modifies all jobs in the flow after creation
# Most flexible - works at flow level and applies uniformly
# flow = update_user_siesta_settings(
#    flow,
#    {
#        "a2s_kpts": [2, 2, 2],
#        "Mesh.Cutoff": "50 Ry",
#        "PAO.BasisSize": "DZP",
#        "xc.functional": "GGA",
#        "xc.authors": "PBE",
#    },
# )


# Run the workflow
results = run_locally(flow, create_folders=True, root_dir="01_li_diffusion")

print(
    "✓ NEB complete: 5 images, vacancy diffusion (2x2x2 kpts, 50 Ry cutoff, DZP basis, PBE)"
)
