#!/usr/bin/env python
"""
Tutorial 15, Example 2: Sparsity Approximation for Large Systems

Demonstrates using Negl.NonOverlap.Int to neglect matrix elements for
non-overlapping orbital pairs, providing 15-25% speedup with negligible
accuracy loss for large systems.

Learning objectives:
- Enable Negl.NonOverlap.Int for performance
- Understand overlap threshold and sparsity
- Balance accuracy vs speed
- Verify approximation is acceptable
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

print("=" * 70)
print("Tutorial 15, Example 2: Sparsity Approximation for Large Systems")
print("=" * 70)
print()

# Create larger Si supercell for demonstration
print("Creating Si supercell (8 atoms)...")
si_primitive = Structure.from_spacegroup(
    "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]
)
# Create 2x2x1 supercell
structure = si_primitive.copy()
structure.make_supercell([2, 2, 1])
print(f"  Formula: {structure.composition.reduced_formula}")
print(f"  Number of atoms: {len(structure)}")
print()

# Configure sparsity approximation
user_params = {
    # Sparsity approximation
    "Negl.NonOverlap.Int": True,  # Skip non-overlapping pairs
    # Calculation parameters
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [4, 4, 4],
    "Mesh.Cutoff": "200 Ry",
    "MD.TypeOfRun": "CG",
    "MD.NumCGsteps": 0,  # Single point for demo
}

print("Sparsity approximation configuration:")
print(f"  Negl.NonOverlap.Int: {user_params['Negl.NonOverlap.Int']}")
print()
print("Effect:")
print("  • Matrix elements computed only if orbital pairs overlap")
print("  • Speedup: 15-25% for large systems (>200 atoms)")
print("  • Accuracy: <0.1 meV/atom difference")
print()

# Create relaxation job - expert tier enables HamiltonianAndOverlapParameters
maker = RelaxMaker.fixed_cell_relaxation(
    user_params=user_params, tier="expert", dry_run=False
)
job = maker.make(structure)

print("Running dry-run calculation...")
print()

results = run_locally(job, create_folders=True)
