#!/usr/bin/env python
"""
Tutorial: Customizing DOS Parameters with Powerups

This tutorial demonstrates how to use powerups to modify DOS calculation
parameters after creating the maker. Powerups provide a flexible way to
update parameters, add convergence settings, or modify calculation options.

Use Case:
---------
When you need to:
- Modify DOS parameters after maker creation
- Apply parameter updates to existing jobs
- Add convergence settings or SCF parameters
- Customize DOS calculations in workflows

Example:
--------
Silicon DOS calculation with powerup-based parameter customization.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import DOSMaker
from atomate2.siesta.powerups import update_user_siesta_settings
from jobflow import run_locally

# Load silicon structure
structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

print("\n" + "=" * 70)
print("Tutorial: Customizing DOS Parameters with Powerups")
print("=" * 70)
print(f"\nStructure: {structure.composition}")
print(f"Number of atoms: {len(structure)}")

# Create basic DOS maker with minimal settings
dos_maker = DOSMaker.dos_calculation(
    dry_run=True,
    user_params={
        "a2s_kpts": [6, 6, 6],  # Initial SCF k-grid
        "Mesh.Cutoff": "250 Ry",  # Initial mesh cutoff
    },
)

# Create job from maker
job = dos_maker.make(structure)

print("\n🔧 Applying powerups to update parameters...")

# Method 1: Update DOS-specific parameters
job = update_user_siesta_settings(
    job,
    {
        # Update SCF parameters
        "Mesh.Cutoff": "350 Ry",  # Increase mesh cutoff
        "a2s_kpts": [8, 8, 8],  # Denser SCF k-grid
        # Update DOS k-grid to denser sampling
        "%block DOS.kgrid.MonkhorstPack": [
            "12 0 0 0.0",
            "0 12 0 0.0",
            "0 0 12 0.0",
        ],
        # Add SCF convergence parameters
        "SCF.DM.Tolerance": "1e-6 eV",  # Tighter convergence
        "SCF.Mixer.Weight": 0.1,  # More conservative mixing
        "SCF.Mixer.History": 5,  # Longer mixing history
        # Add electronic temperature for better convergence
        "ElectronicTemperature": "100 K",  # Lower temperature
    },
)

# Method 2: Chain multiple powerup updates
print("\n🔧 Applying additional powerup for occupation function...")

job = update_user_siesta_settings(
    job,
    {
        "OccupationFunction": "MP",  # Methfessel-Paxton smearing
        "OccupationMPOrder": 1,  # First-order MP
    },
)

print("\n✓ Added Methfessel-Paxton occupation function")

# Run the job
response = run_locally(job, create_folders=True)
