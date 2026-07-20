#!/usr/bin/env python
"""Phonon calculation with custom parameters for relaxation and forces.

Key strategy:
- Relaxation: Moderate parameters (faster, finds geometry)
- Force calculations: Tight parameters (accurate forces → better phonons)

Runtime: ~1-2 hours
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.jobs.core import StaticMaker, LuaMaker
from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker

# Load structure
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")
# structure = Structure.from_file("Si.cif")

# ============================================================================
# RELAXATION PARAMETERS (moderate - just needs good geometry)
# ============================================================================
relax_params = {
    "PAO.BasisSize": "DZP",
    # "PAO.EnergyShift": "0.05 Ry",  # 272 meV
    "a2s_kpts": [3, 3, 3],
    "Mesh.Cutoff": "300 Ry",
    "DM.Tolerance": 1e-5,
    "MD.MaxForceTol": "0.01 eV/Ang",
}

# ============================================================================
# FORCE PARAMETERS (tight - critical for accurate phonons!)
# ============================================================================
force_params = {
    "PAO.BasisSize": "DZP",
    # "PAO.EnergyShift": "0.01 Ry",  # 136 meV - tighter basis
    "a2s_kpts": [2, 2, 2],  # Denser k-points
    "Mesh.Cutoff": "450 Ry",  # Higher cutoff
    "DM.Tolerance": 1e-6,  # Tighter SCF
}

# ============================================================================
# CREATE MAKERS WITH CUSTOM PARAMETERS
# ============================================================================

# Create makers with parameters directly in user_params
# relax_maker = RelaxMaker.variable_cell_relaxation(
#    use_custodian=True,  # Enable error handling
#    custodian_max_errors=10,
#    user_params=relax_params,
# )

relax_maker = LuaMaker.variable_cell_relaxation(
    use_custodian=True,  # Enable error handling
    custodian_max_errors=10,
    user_params=relax_params,
)


static_maker = StaticMaker.scf(
    use_custodian=True,  # Enable error handling
    user_params=force_params,
)

# Create phonon workflow
maker = SiestaPhononFlowMaker(
    relax_maker=relax_maker,  # Custom relaxation with custodian
    static_maker=static_maker,  # Custom force calculations with custodian
    # Supercell size:
    # min_length=12.0,  # Auto-generate (larger than basic example)
    # OR explicit:
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],  # 2×2×2
    prefer_90_degrees=True,
    # Displacement
    displacement=0.01,
    use_symmetry=True,
    # Q-point sampling
    mesh=(50, 50, 50),  # Denser than basic example
    # Thermal properties
    create_thermal_properties=True,
    t_min=0,
    t_max=1000,
    t_step=10,
)

print("Relaxation: DZP, 0.01 Ry, [6,6,6], 300 Ry")
print("Forces:     DZP, 0.005 Ry, [8,8,8], 400 Ry (tighter!)")

# Run workflow
flow = maker.make(structure)
results = run_locally(
    flow, create_folders=True, ensure_success=True, root_dir="01_standard"
)

print(
    "\n✓ Complete! Check: phonon_bands.png, phonon_dos.png, "
    "thermal_properties.png, phonon_summary.txt"
)
