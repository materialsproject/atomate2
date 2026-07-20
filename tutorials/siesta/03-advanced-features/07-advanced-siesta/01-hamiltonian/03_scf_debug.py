#!/usr/bin/env python
"""
Tutorial 15, Example 3: SCF Debugging with Extra Information

Demonstrates using SCF.WriteExtra to enable verbose SCF output including
eigenvalue spectrum, Mulliken populations, and charge transfer analysis
for debugging convergence issues.

Learning objectives:
- Enable SCF.WriteExtra for detailed output
- Understand eigenvalue spectrum evolution
- Analyze Mulliken populations during SCF
- Debug convergence problems
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

print("=" * 70)
print("Tutorial 15, Example 3: SCF Debugging with Extra Information")
print("=" * 70)
print()

# Create Si structure
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Configure SCF debugging
user_params = {
    # SCF debugging
    "SCF.WriteExtra": True,  # Verbose SCF output
    # SCF parameters (intentionally difficult for demo)
    "SCF.MixingWeight": 0.1,  # Slower, more stable
    "SCF.Mixer.History": 8,  # More Pulay history
    # Calculation parameters
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [6, 6, 6],
    "Mesh.Cutoff": "300 Ry",
}

print("SCF debugging configuration:")
print(f"  SCF.WriteExtra: {user_params['SCF.WriteExtra']}")
print(f"  SCF.MixingWeight: {user_params['SCF.MixingWeight']}")
print(f"  SCF.Mixer.History: {user_params['SCF.Mixer.History']}")
print()

print("Additional SCF output will include:")
print("  • Eigenvalue spectrum at each iteration")
print("  • Mulliken population analysis")
print("  • Charge transfer between atoms")
print("  • Band structure evolution (if applicable)")
print()

# Create static calculation with debugging - expert tier enables HamiltonianAndOverlapParameters
maker = StaticMaker.scf(user_params=user_params, tier="expert", dry_run=False)
job = maker.make(structure)

print("Running dry-run calculation...")
print("In a real calculation, check siesta.out for extra SCF details")
print()

results = run_locally(job, create_folders=True)
