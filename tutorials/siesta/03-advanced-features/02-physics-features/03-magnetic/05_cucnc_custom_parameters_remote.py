#!/usr/bin/env python
"""Magnetic CuNCN surface with DFT+U and automatic DM.InitSpin generation.

This example demonstrates:
1. Automatic DM.InitSpin generation from structure.magmom
2. DFT+U for Cu 3d orbitals (strongly correlated)
3. Custom configuration for slab + adsorbate calculations
4. Ferromagnetic ordering for Cu atoms

Key features:
- Zero-code magnetic moment handling
- DFT+U parameters for Cu
- Clean DM.InitSpin output (+0.5 instead of +0.500000000000000)
- Production-ready for HPC clusters
"""

from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.core import StaticSetGenerator
from atomate2.siesta.powerups import siesta_to_pymatgen
from jobflow import run_locally

from atomate2.siesta.sets.utils import get_default_initial_magnetic_moments

# ============================================================================
# CONFIGURATION - Define all parameters here
# ============================================================================
CONFIG = {
    # Slab+adsorbate calculation parameters
    "slab_params": {
        "a2s_kpts": [4, 4, 1],  # k-point mesh (dense in-plane, Γ in z)
        "Mesh.Cutoff": "200 Ry",  # Real-space grid cutoff
        "PAO.BasisSize": "TZDP",  # Double-zeta polarized basis
        "OccupationFunction": "MP",  # Methfessel-Paxton smearing
        "ElectronicTemperature": "300 K",
        "SCF.Mixer.Weight": 0.01,  # Mixing weight for SCF
        "SCF.Mixer.Method": "Pulay",
        "SCF.DM.Tolerance": 1e-4,
        "Spin": "polarized",
        # Magnetic ordering (optional - auto-detects from structure.magmom)
        # Options: "antiferromagnetic"/"AFM" (default), "ferromagnetic"/"FM", "custom"
        # "a2s_magnetic_ordering": "ferromagnetic",  # Cu surface is FM
        "a2s_dm_init_spin_format": "sign_only",  # ← NEW! Write just +/- signs
        # DFT+U parameters for Cu 3d orbitals
        "DFTU.ProjectorGenerationMethod": 2,
        "DFTU.CutoffNorm": 0.9,
        "DFTU.FirstIteration": "T",
        "DFTU.ThresholdTol": 0.01,
        "DFTU.PopTol": 0.001,
        "%block DFTU.Proj": [
            "Cu 1                 # label, number of l-shells with U",
            "n=3 2                # n=3 (3d), l=2 (d-shell)",
            "4.0 0.0              # U (eV), J (eV)  -> Ueff = 4 eV",
            "0.0 0.0              # rc, omega (0 0 => use CutoffNorm and default ω)",
        ],
    },
    # Isolated molecule parameters
    "molecule_params": {
        "a2s_kpts": [1, 1, 1],  # Γ-point only for molecules
        "Mesh.Cutoff": "300 Ry",  # Higher cutoff for accuracy
        "PAO.BasisSize": "TZDP",
        "OccupationFunction": "FD",  # Fermi-Dirac (no smearing)
        "ElectronicTemperature": "25 K",  # Low temperature
        "SCF.Mixer.Weight": 0.1,
        "SCF.DM.Tolerance": 1e-5,
    },
    # Grid scanning parameters
    "grid_size": (2, 2),
    "height": 1.0,
    # Error handling
    "use_custodian": True,
    "custodian_max_errors": 10,
}

# Alternative configurations:
#
# # High-accuracy configuration
# CONFIG_HIGH_ACCURACY = {
#     "slab_params": {
#         "a2s_kpts": [6, 6, 1],
#         "Mesh.Cutoff": "400 Ry",
#         "PAO.BasisSize": "TZP",
#         "OccupationFunction": "MP",
#         "ElectronicTemperature": "300 K",
#         "SCF.Mixer.Weight": 0.005,
#         "SCF.DM.Tolerance": 1e-6,
#     },
#     "molecule_params": {
#         "a2s_kpts": [1, 1, 1],
#         "Mesh.Cutoff": "500 Ry",
#         "PAO.BasisSize": "TZP",
#         "OccupationFunction": "FD",
#         "ElectronicTemperature": "25 K",
#         "SCF.DM.Tolerance": 1e-7,
#     },
#     "grid_size": (5, 5),
#     "height": 2.0,
#     "use_custodian": True,
#     "custodian_max_errors": 15,
# }
#
# # Fast screening configuration
# CONFIG_FAST = {
#     "slab_params": {
#         "a2s_kpts": [2, 2, 1],
#         "Mesh.Cutoff": "150 Ry",
#         "PAO.BasisSize": "SZP",
#         "OccupationFunction": "MP",
#         "ElectronicTemperature": "500 K",
#         "SCF.Mixer.Weight": 0.02,
#     },
#     "molecule_params": {
#         "a2s_kpts": [1, 1, 1],
#         "Mesh.Cutoff": "200 Ry",
#         "PAO.BasisSize": "DZP",
#         "OccupationFunction": "FD",
#         "ElectronicTemperature": "50 K",
#     },
#     "grid_size": (2, 2),
#     "height": 2.0,
#     "use_custodian": False,
#     "custodian_max_errors": 5,
# }

# ============================================================================
# WORKFLOW SETUP
# ============================================================================

slab = siesta_to_pymatgen("../../../00-structures/CuNCN-supercell-3l-101-121.xsf")


# Try automatic magnetic moment detection
print(f"\n{'='*70}")
print("Magnetic Moment Setup")
print(f"{'='*70}")

magmoms = get_default_initial_magnetic_moments(slab)
print(f"Automatic detection: {magmoms}")

# Cu is now auto-detected (Z=29 added to MAGNETIC_ELEMENTS)
# Add magmom property to structure
slab.add_site_property("magmom", magmoms)
"""
n_magnetic = sum(1 for m in magmoms if abs(m) > 1e-6)
n_cu = sum(1 for s in slab if s.specie.symbol == "Cu")
print(f"✓ Set magnetic moments for {len(slab)} atoms:")
print(f"  - {n_cu} Cu atoms: 0.6 μB each")
print(f"  - {len(slab) - n_cu} non-magnetic atoms: 0.0 μB")
print(f"✓ DM.InitSpin will be auto-generated from structure.magmom!")
print(f"{'='*70}\n")
"""

molecule = siesta_to_pymatgen("../../../00-structures/H2.xsf", as_molecule=True)


# Create input generators with custom parameters
slab_generator = StaticSetGenerator(user_params=CONFIG["slab_params"])
molecule_generator = StaticSetGenerator(user_params=CONFIG["molecule_params"])

# Create makers
slab_maker = StaticMaker(
    input_set_generator=slab_generator,
    use_custodian=CONFIG["use_custodian"],
    custodian_max_errors=CONFIG["custodian_max_errors"],
)

# Molecule maker - explicitly set to 4 cores
adsorbate_maker = StaticMaker(
    input_set_generator=molecule_generator,
    use_custodian=CONFIG["use_custodian"],
)
# Set resources after creating the maker
adsorbate_maker.config_dict = {
    "manager_config": {
        "resources": {
            "mem_per_cpu": "4G",
            "nodes": 1,
            "ntasks_per_node": 4,  # ← 4 cores for molecule!
            "cpus_per_task": 1,
            "time": "24:00:00",
        }
    }
}


# Print configuration
print("Configuration:")
print("\nSlab parameters:")
for key, value in CONFIG["slab_params"].items():
    print(f"  {key}: {value}")
print("\nMolecule parameters:")
for key, value in CONFIG["molecule_params"].items():
    print(f"  {key}: {value}")
print(f"\nGrid size: {CONFIG['grid_size']}")
print(f"Custodian: {'enabled' if CONFIG['use_custodian'] else 'disabled'}")

# Create workflow
flow = AdsorptionScanFlowMaker(
    dry_run=True,
    name="Final-CuCSN-7l-100-O2-Scan",
    grid_size=CONFIG["grid_size"],
    height=CONFIG["height"],
    slab_static_maker=slab_maker,
    adsorbate_static_maker=adsorbate_maker,
)

# Run workflow
workflow = flow.make(slab, molecule)
results = run_locally(workflow, create_folders=True)

# ============================================================================
# FIX RESOURCES - Update each job's config AFTER creating workflow
# ============================================================================
print("\nUpdating resources for jobs:")
for job in workflow:
    job_name = job.name

    # Set 4 cores for molecule/adsorbate jobs
    if "molecule" in job_name.lower() or "adsorbate" in job_name.lower():
        job.update_config(
            {
                "manager_config": {
                    "resources": {
                        "mem_per_cpu": "4G",
                        "nodes": 1,
                        "ntasks_per_node": 4,  # ← 4 cores for molecule!
                        "cpus_per_task": 1,
                        "time": "24:00:00",
                    }
                }
            }
        )
        print(f"  {job_name}: 4 cores")
    else:
        print(f"  {job_name}: 24 cores (default)")

print("\n✓ Adsorption scan complete!")
print("\nTo change calculation parameters:")
print("  1. Modify the CONFIG dictionary at the top of this file")
print("  2. Available parameters: kpts, Mesh.Cutoff, PAO.BasisSize,")
print("     OccupationFunction, ElectronicTemperature, SCF.Mixer.Weight, etc.")
print("  3. See commented examples for high-accuracy and fast-screening configs")

"""
results = submit_flow(
    workflow,
    # project="atomate2siesta",
    # worker="agustina_worker",
    project="alberto",
    worker="cesga_worker",
    resources={
        # "partition": "RES", # for Agustina
        # "account": "icn2100", # for Agustina
        # "mem": "500GB",
        "mem_per_cpu": "4G",  # For cesga
        "nodes": 1,
        "ntasks_per_node": 24,
        "cpus_per_task": 1,
        # "ntasks": 24,
        "time": "24:00:00",
    },
)
"""
