#!/usr/bin/env python
"""Quick elastic test with minimal settings."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.elastic import ElasticFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

flow = ElasticFlowMaker()
workflow = flow.make(structure)
workflow = update_user_siesta_settings(
    workflow, {"PAO.BasisSize": "DZ", "a2s_kpts": [3, 3, 3], "Mesh.Cutoff": "200 Ry"}
)
results = run_locally(workflow, create_folders=True, root_dir="01_minimal")

print("✓ Quick elastic test complete")
