#!/usr/bin/env python
"""Magnetic calculation with DFT+U for strongly correlated systems.

This example combines automatic DM.InitSpin generation with DFT+U
for transition metal oxides. Perfect for Cu, Ni, Co, Fe oxides.
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

# Create CuO structure
lattice = Lattice.monoclinic(4.68, 3.42, 5.13, 99.5)
structure = Structure(
    lattice,
    ["Cu", "Cu", "O", "O"],
    [[0.25, 0.25, 0.0], [0.75, 0.75, 0.5], [0.0, 0.42, 0.25], [0.5, 0.58, 0.75]],
)

# Set magnetic moments on Cu atoms (Cu2+ typically ~0.5-0.7 μB with DFT+U)
structure.add_site_property("magmom", [0.6, 0.6, 0.0, 0.0])

print(f"Structure: {structure.composition}")
print(f"Magnetic moments: {structure.site_properties['magmom']}")

# Create maker with DFT+U for Cu 3d orbitals
maker = StaticMaker.scf(
    dry_run=True,
    user_params={
        # Magnetic settings
        "Spin": "polarized",
        "a2s_magnetic_ordering": "AFM",  # CuO is antiferromagnetic
        # Basic parameters
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP",
        # DFT+U parameters for Cu 3d
        "DFTU.ProjectorGenerationMethod": 2,
        "DFTU.CutoffNorm": 0.9,
        "DFTU.FirstIteration": "T",
        "DFTU.ThresholdTol": 0.01,
        "DFTU.PopTol": 0.001,
        "%block DFTU.Proj": [
            "Cu 1                 # label, number of l-shells with U",
            "n=3 2                # n=3 (3d), l=2 (d-shell)",
            "7.0 0.0              # U (eV), J (eV)  -> Ueff = 7 eV for CuO",
            "0.0 0.0              # rc, omega (0 0 => use defaults)",
        ],
    },
)

# Run calculation
job = maker.make(structure)
response = run_locally(job, create_folders=True)

print("\n✓ Calculation complete!")
print("✓ DM.InitSpin + DFT+U enabled")
print("  AFM ordering on Cu atoms")
print("  U = 7 eV on Cu 3d orbitals")
