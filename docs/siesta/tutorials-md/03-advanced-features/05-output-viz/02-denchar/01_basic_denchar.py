#!/usr/bin/env python
"""
Tutorial 11, Example 1: Basic Denchar Output

Demonstrates enabling denchar with default grid resolution.

This example shows how to configure SIESTA to output denchar files
for post-processing and visualization.

Learning objectives:
- Enable denchar file output
- Use default grid resolution (50x50x50)
- Understand denchar output files
- Basic denchar workflow
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

print("=" * 70)
print("Tutorial 11, Example 1: Basic Denchar Output")
print("=" * 70)
print()

# Create a benzene molecule
print("Creating C6H6 (benzene) structure...")
lattice = Lattice.cubic(15.0)  # Large box for molecule
structure = Structure(
    lattice,
    ["C"] * 6 + ["H"] * 6,
    [
        # Carbon ring
        [0.5, 0.5, 0.5],
        [0.52, 0.52, 0.5],
        [0.52, 0.54, 0.5],
        [0.5, 0.56, 0.5],
        [0.48, 0.54, 0.5],
        [0.48, 0.52, 0.5],
        # Hydrogens
        [0.5, 0.48, 0.5],
        [0.53, 0.51, 0.5],
        [0.53, 0.55, 0.5],
        [0.5, 0.58, 0.5],
        [0.47, 0.55, 0.5],
        [0.47, 0.51, 0.5],
    ],
    coords_are_cartesian=False,
)
print(f"  Formula: {structure.composition.reduced_formula}")
print(f"  Atoms: {len(structure)}")
print("  Geometry: Planar aromatic ring")
print()

# Configure denchar output using SIESTA FDF names
print("Configuring denchar output...")
user_params = {
    # Enable denchar
    "Write.Denchar": True,
    # Default grid resolution (50x50x50 is adequate for preview)
    # Note: NumberPointsX/Y/Z use default values from dataclass
    # Calculation parameters
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [1, 1, 1],  # Gamma point for molecule
    "Mesh.Cutoff": "200 Ry",
}
# Create static calculation job
maker = StaticMaker.scf(user_params=user_params, dry_run=True)
job = maker.make(structure)
print(f"  Job name: {job.name}")

# Uncomment to run:
results = run_locally(job, create_folders=True)
