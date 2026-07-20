#!/usr/bin/env python
"""Custom calculation parameters without using tier presets.

This tutorial demonstrates how to define calculation parameters explicitly
without relying on tier presets. This gives you maximum control and allows
you to create completely custom parameter sets for specific materials or
calculation types.

Use this approach when:
- Tier presets don't match your needs
- You need fine-grained control over all parameters
- You're doing method development or testing
- You want to reproduce literature parameters exactly
"""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

# ============================================================================
# CONFIGURATION - All parameters defined explicitly
# ============================================================================
CONFIG = {
    # k-point sampling
    "a2s_kpts": [8, 8, 8],  # Dense Monkhorst-Pack grid
    # Real-space mesh
    "Mesh.Cutoff": "400 Ry",  # High cutoff for accuracy
    # Basis set
    "PAO.BasisSize": "TZP",  # Triple-zeta polarized
    # SCF convergence
    "SCF.Mixer.Method": "Pulay",
    "SCF.Mixer.Weight": 0.05,
    "SCF.Mixer.History": 8,
    "SCF.DM.Tolerance": 1e-6,
    "MaxSCFIterations": 200,
    # Electronic structure
    "OccupationFunction": "FD",
    "ElectronicTemperature": "100 K",
    # Geometry optimization
    "MD.MaxForceTol": "0.01 eV/Ang",
    "MD.MaxCGDispl": "0.1 Ang",
    "MD.MaxStressTol": "0.5 GPa",
    # Exchange-correlation
    "XC.Functional": "GGA",
    "XC.Authors": "PBE",
}

# Alternative configurations for different scenarios:
#
# # Metallic system (needs occupation smearing)
# CONFIG_METAL = {
#     "a2s_kpts": [12, 12, 12],
#     "Mesh.Cutoff": "300 Ry",
#     "PAO.BasisSize": "DZP",
#     "OccupationFunction": "MP",      # Methfessel-Paxton
#     "OccupationMPOrder": 1,
#     "ElectronicTemperature": "300 K",
#     "SCF.Mixer.Weight": 0.01,
#     "SCF.Mixer.Method": "Pulay",
#     "SCF.DM.Tolerance": 1e-5,
#     "MD.MaxForceTol": "0.02 eV/Ang",
# }
#
# # Large system (>100 atoms) - optimized for speed
# CONFIG_LARGE = {
#     "a2s_kpts": [2, 2, 2],
#     "Mesh.Cutoff": "200 Ry",
#     "PAO.BasisSize": "DZP",
#     "SolutionMethod": "OrderN",      # Linear-scaling
#     "ON.MaximumIterations": 1000,
#     "ON.functional": "Kim",
#     "OccupationFunction": "FD",
#     "ElectronicTemperature": "100 K",
#     "SCF.Mixer.Weight": 0.1,
#     "MD.MaxForceTol": "0.05 eV/Ang",
# }
#
# # Magnetic system
# CONFIG_MAGNETIC = {
#     "a2s_kpts": [8, 8, 8],
#     "Mesh.Cutoff": "400 Ry",
#     "PAO.BasisSize": "DZP",
#     "spin": "polarized",
#     "OccupationFunction": "FD",
#     "ElectronicTemperature": "50 K",
#     "SCF.Mixer.Weight": 0.002,       # Very slow mixing for spin
#     "SCF.Mixer.Method": "Pulay",
#     "SCF.Mixer.History": 10,
#     "SCF.DM.Tolerance": 1e-6,
#     "MD.MaxForceTol": "0.01 eV/Ang",
# }
#
# # Ultra-high accuracy
# CONFIG_HIGH_ACCURACY = {
#     "a2s_kpts": [16, 16, 16],
#     "Mesh.Cutoff": "600 Ry",
#     "PAO.BasisSize": "TZP",
#     "PAO.EnergyShift": "0.001 Ry",
#     "PAO.SplitNorm": 0.10,
#     "OccupationFunction": "FD",
#     "ElectronicTemperature": "10 K",
#     "SCF.Mixer.Weight": 0.02,
#     "SCF.DM.Tolerance": 1e-7,
#     "SCF.H.Tolerance": 1e-5,
#     "MD.MaxForceTol": "0.001 eV/Ang",
#     "MD.MaxCGDispl": "0.01 Ang",
# }

# ============================================================================
# WORKFLOW SETUP
# ============================================================================

# Load structure
bulk = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Create relaxation maker - pass user_params directly!
maker = RelaxMaker.fixed_cell_relaxation(
    user_params=CONFIG,  # Pass parameters directly to the class method
    use_custodian=True,
    custodian_max_errors=10,
)

# Print configuration
print("Custom Calculation Parameters:")
print("=" * 60)
for key, value in CONFIG.items():
    print(f"  {key:<30} : {value}")
print("=" * 60)

# Run calculation
job = maker.make(bulk)
results = run_locally(job, create_folders=True)

print("\n✓ Calculation complete!")
print("\nParameter categories used:")
print("  • k-point sampling: kpts")
print("  • Real-space mesh: Mesh.Cutoff")
print("  • Basis set: PAO.BasisSize, PAO.EnergyShift, PAO.SplitNorm")
print("  • SCF: SCF.Mixer.*, SCF.DM.Tolerance, MaxSCFIterations")
print("  • Electronic: OccupationFunction, ElectronicTemperature")
print("  • Geometry: MD.MaxForceTol, MD.MaxCGDispl, MD.MaxStressTol")
print("  • XC functional: XC.Functional, XC.Authors")
print("\nSee commented examples for:")
print("  - Metallic systems (occupation smearing)")
print("  - Large systems (linear-scaling)")
print("  - Magnetic systems (spin polarization)")
print("  - Ultra-high accuracy settings")
