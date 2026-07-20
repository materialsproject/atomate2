#!/usr/bin/env python
"""
EOS with Custom Mesh Cutoff via user_params.

This tutorial shows how to customize Mesh.Cutoff when running EOS calculations
using user_params in the RelaxMaker.

Key concepts:
- user_params: Dictionary passed to input set generator via RelaxMaker
- eos_relax_maker: Maker used for energy calculations at each volume
- initial_relax_maker: Optional maker for initial structure relaxation
- Applies to ALL calculations in the workflow
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.eos import SiestaEosFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.utils import get_default_initial_magnetic_moments

# Load structure
# structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")
structure = Structure.from_file("../../../00-structures/MoS2.cif")

# Add magnetic moments (only if structure has magnetic elements)
magmoms = get_default_initial_magnetic_moments(structure)
if magmoms is not None:
    structure.add_site_property("magmom", magmoms)
    print(f"Added magnetic moments: {magmoms}")
else:
    print("No magnetic elements detected - skipping magnetic moment initialization")

# Method : Different parameters for initial vs EOS relaxations
# ==============================================================
print("\nMethod : Different Mesh.Cutoff for initial vs EOS relaxations")

# Initial relaxation with lower accuracy (faster)
initial_maker = RelaxMaker.variable_cell_relaxation(
    user_params={
        "mesh.cutoff": 300,  # Lower for speed
        "PAO.BasisSize": "DZ",  # Smaller basis
        "a2s_kpts": [4, 4, 4],
        "Spin": "polarized",  # Enable spin polarization
        "a2s_magnetic_ordering": "antiferromagnetic",  # AFM ordering
    },
)

# EOS relaxations with higher accuracy
eos_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "mesh.cutoff": 500,  # Higher for accuracy
        "PAO.BasisSize": "DZP",  # Larger basis
        "a2s_kpts": [6, 6, 6],
        "Spin": "polarized",  # Enable spin polarization
        "a2s_magnetic_ordering": "antiferromagnetic",  # AFM ordering
    },
)

# Create EOS workflow with both makers
maker2 = SiestaEosFlowMaker(
    dry_run=True,
    initial_relax_maker=initial_maker,  # Initial: fast settings
    eos_relax_maker=eos_maker,  # EOS points: accurate settings
    linear_strain=(-0.04, 0.04),
    number_of_frames=5,
)

workflow2 = maker2.make(structure)
results2 = run_locally(
    workflow2, create_folders=True, root_dir="03_mesh_cutoff_user_params"
)

print("✓ Two-stage EOS complete!")
print("  - Initial relaxation: mesh_cutoff=300 Ry, basis=DZ")
print("  - EOS calculations: mesh_cutoff=500 Ry, basis=DZP")
print("  - This saves time while maintaining accuracy")
