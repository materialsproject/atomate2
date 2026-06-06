#!/usr/bin/env python
"""Example 5: Magnetic Calculations and DFT+U.

Demonstrates how to set up:
1. Spin-polarized calculations
2. DFT+U for correlated systems
3. Combining magnetism + DFT+U
"""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

print("=" * 80)
print("Example 5: Magnetic Calculations and DFT+U")
print("=" * 80)
print()

# ==============================================================================
# Part 1: Simple Magnetic Calculation (Spin-Polarized)
# ==============================================================================
print("Part 1: Spin-Polarized Calculation")
print("-" * 80)

# BCC Fe structure
fe = Structure.from_spacegroup("Im-3m", Lattice.cubic(2.87), ["Fe"], [[0, 0, 0]])

flow_magnetic = RecipeBook.band_structure_workflow(
    fe,
    auto_params=False,  # Disable auto-detection
    user_params={
        "Spin": "polarized",  # Enable spin polarization
        "a2s_magnetic_ordering": "FM",  # Ferromagnetic ordering
        "OccupationFunction": "MP",  # Methfessel-Paxton smearing
        "ElectronicTemperature": "300 K",  # Electronic temperature
    },
    dry_run=True,
)

print("✅ Created ferromagnetic Fe calculation")
print("   Parameters: Spin=polarized, FM ordering, MP smearing")
print()

results1 = run_locally(flow_magnetic, create_folders=True)
print("✅ Input files generated for magnetic calculation")
print()

# ==============================================================================
# Part 2: DFT+U Calculation
# ==============================================================================
print("\nPart 2: DFT+U for Correlated System")
print("-" * 80)

# NiO structure (antiferromagnetic insulator)
nio = Structure.from_spacegroup(
    "Fm-3m", Lattice.cubic(4.177), ["Ni", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
)

flow_dftu = RecipeBook.elastic_constants_workflow(
    nio,
    auto_params=False,  # Disable auto-detection
    preset="magnetic_correlated",  # Preset for magnetic + correlated systems
    user_params={
        "DFTU.ProjectorGenerationMethod": 2,  # 1=Hydrogenic, 2=Bessel, 3=Filtered
        "DFTU.CutoffNorm": 0.9,
        "DFTU.FirstIteration": "T",
        "%block DFTU.Proj": [
            "Ni 1                 # element, number of l-shells with U",
            "n=3 2                # n=3 (3d), l=2 (d-shell)",
            "5.3 0.0              # U (eV), J (eV)",
            "0.0 0.0              # rc, omega (0 0 => defaults)",
        ],
        "a2s_kpts": [6, 6, 6],
    },
    dry_run=True,
)

print("✅ Created NiO calculation with DFT+U")
print("   U(Ni-d) = 5.3 eV, Projector Method = 2 (Bessel)")
print()

results2 = run_locally(flow_dftu, create_folders=True)
print("✅ Input files generated for DFT+U calculation")
print()

# ==============================================================================
# Summary
# ==============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("Magnetic calculations:")
print("  Spin = 'polarized'")
print("  a2s_magnetic_ordering = 'FM' (ferromagnetic) or 'AFM' (antiferromagnetic)")
print("  OccupationFunction = 'MP' or 'FD' (smearing for metals)")
print()
print("DFT+U parameters (SIESTA syntax):")
print("  DFTU.ProjectorGenerationMethod = 1 (Hydrogenic), 2 (Bessel), 3 (Filtered)")
print("  %block DFTU.Proj:")
print("    Element N_shells")
print("    n=X l_value")
print("    U J")
print("    rc omega")
print("  %endblock")
print()
print("Example:")
print("  '%block DFTU.Proj': [")
print("      'Cu 1            # element, number of l-shells',")
print("      'n=3 2           # n=3 (3d shell), l=2 (d orbital)',")
print("      '7.0 0.0         # U=7 eV, J=0',")
print("      '0.0 0.0         # use defaults for rc, omega',")
print("  ]")
print()
print("Common U values (eV):")
print("  Ni 3d: 5-6")
print("  Cu 3d: 6-8")
print("  Fe 3d: 4-5")
print("  Co 3d: 4-5")
print("  (Values vary by functional and system)")
print()
