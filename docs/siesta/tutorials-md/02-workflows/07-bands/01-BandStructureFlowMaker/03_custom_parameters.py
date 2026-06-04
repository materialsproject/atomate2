#!/usr/bin/env python
"""Band structure workflow with custom parameters.

This tutorial demonstrates:
- Custom basis set (TZP for higher accuracy)
- Custom mesh cutoff
- Custom k-point sampling for SCF
- Custom energy range for plotting
- Custodian error handling

Use these customizations for:
- Publication-quality results
- Difficult convergence cases
- Materials requiring higher accuracy

Runtime: ~45-60 minutes (higher accuracy settings)
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.bands import BandStructureFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker, BandStructureMaker

# Load silicon primitive cell
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# ============================================================================
# CUSTOM MAKERS WITH HIGH-ACCURACY PARAMETERS
# ============================================================================

# Relaxation: Tight convergence for accurate geometry
relax_maker = RelaxMaker.variable_cell_relaxation(
    use_custodian=True,
    custodian_max_errors=10,
    user_params={
        "PAO.BasisSize": "TZP",  # Triple-zeta polarized
        "Mesh.Cutoff": "400 Ry",  # High cutoff
        "a2s_kpts": [8, 8, 8],  # Dense k-grid
        "MD.MaxForceTol": "0.01 eV/Ang",  # Tight force tolerance
        "MD.MaxStressTol": "0.1 GPa",  # Tight stress tolerance
    },
)

# SCF: Dense k-grid for accurate band edges
scf_maker = StaticMaker.scf(
    use_custodian=True,
    custodian_max_errors=10,
    user_params={
        "PAO.BasisSize": "TZP",
        "Mesh.Cutoff": "400 Ry",
        "a2s_kpts": [12, 12, 12],  # Very dense for accurate Fermi level
        "DM.Tolerance": 1e-6,  # Tight SCF convergence
    },
)

# Bands: Same basis as SCF for consistency
bands_maker = BandStructureMaker.bandstructure_calculation(
    use_custodian=True,
    user_params={
        "PAO.BasisSize": "TZP",
        "Mesh.Cutoff": "400 Ry",
        "WriteBands": "true",  # Required to generate .bands file for plotting
    },
)

# ============================================================================
# CREATE WORKFLOW WITH CUSTOM MAKERS
# ============================================================================

maker = BandStructureFlowMaker(
    relax_maker=relax_maker,
    scf_maker=scf_maker,
    bands_maker=bands_maker,
    plot_bands=True,
    energy_range=(-8, 8),  # Wider energy range for plot (eV from Fermi)
)

# Generate and run workflow
flow = maker.make(structure)
results = run_locally(flow, create_folders=True, root_dir="03_custom_parameters")

print("\n" + "=" * 60)
print("High-Accuracy Band Structure Complete!")
print("=" * 60)
print("\nSettings used:")
print("  - Basis: TZP (Triple-zeta polarized)")
print("  - Mesh cutoff: 400 Ry")
print("  - SCF k-points: 12x12x12")
print("  - Energy range: -8 to +8 eV")
print("\nThese settings provide publication-quality results.")
