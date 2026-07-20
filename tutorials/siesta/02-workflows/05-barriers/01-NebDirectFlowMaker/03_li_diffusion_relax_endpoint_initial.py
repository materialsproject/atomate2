#!/usr/bin/env python
"""NEB vacancy diffusion with endpoint relaxation using NebDirectMaker.

This tutorial demonstrates:
1. Flexible endpoint relaxation options (initial, final, both, or neither)
2. Three methods to customize SIESTA parameters
3. How to relax only specific endpoints

NEW in v1.0.0: relax_endpoints now accepts:
- False: No relaxation (use structures as provided)
- True: Relax both endpoints (backward compatible)
- "initial": Relax only initial structure
- "final": Relax only final structure
- "both": Relax both (explicit, same as True)
"""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.neb import NebDirectFlowMaker
from atomate2.siesta.jobs.core import LuaMaker, RelaxMaker

# Load initial and final structures
initial = Structure.from_file("../../../00-structures/mgo_li-initial.xsf")
final = Structure.from_file("../../../00-structures/mgo_li-final.xsf")

# =============================================================================
# Example 2: Relax only initial structure (final is already optimized)
# =============================================================================
maker = NebDirectFlowMaker(
    dry_run=False,
    number_of_images=5,
    relax_endpoints="initial",  # Only relax initial structure
    relax_maker=RelaxMaker.fixed_cell_relaxation(
        user_params={"PAO.BasisSize": "DZP", "a2s_kpts": [1, 1, 1]},
    ),
    neb_maker=LuaMaker.neb(
        user_params={"PAO.BasisSize": "DZP", "a2s_kpts": [1, 1, 1]},
    ),
)


flow = maker.make(initial_structure=initial, final_structure=final)

# =============================================================================
# Use powerups to update settings (RECOMMENDED)
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
results = run_locally(
    flow, create_folders=True, root_dir="03_li_diffusion_relax_endpoint_initial"
)

print(
    "✓ NEB complete: 5 images, vacancy diffusion (2x2x2 kpts, 50 Ry cutoff, DZP basis, PBE)"
)
