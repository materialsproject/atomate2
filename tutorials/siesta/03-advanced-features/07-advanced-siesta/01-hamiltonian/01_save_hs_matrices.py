#!/usr/bin/env python
"""
Tutorial 15, Example 1: Save Hamiltonian and Overlap Matrices

Demonstrates saving H and S matrices to .HS files for post-processing workflows
like band structure analysis, transport calculations, and wannier90 interface.

Learning objectives:
- Enable SaveHS for matrix output
- Understand .HS file format and contents
- Use .HS files for post-processing (DOS, bands, transport)
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

print("=" * 70)
print("Tutorial 15, Example 1: Save Hamiltonian and Overlap Matrices")
print("=" * 70)
print()

# Create Si structure
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Configure matrix saving
user_params = {
    # Matrix output
    "SaveHS": True,  # Save H and S matrices (default=True)
    # Calculation parameters
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [8, 8, 8],  # Dense k-sampling for accurate matrices
    "Mesh.Cutoff": "300 Ry",
}

print("Hamiltonian matrix configuration:")
print(f"  SaveHS: {user_params['SaveHS']}")
print("  Output file: systemLabel.HS (binary format)")
print()

print("Matrix contents:")
print("  • Hamiltonian matrix (H) - Energy operator")
print("  • Overlap matrix (S) - Basis function overlaps")
print("  • Sparse format (COO) - Only non-zero elements")
print("  • System information - Atoms, orbitals, cell vectors")
print()

# Create static calculation - expert tier enables HamiltonianAndOverlapParameters
maker = StaticMaker.scf(user_params=user_params, tier="expert", dry_run=True)
job = maker.make(structure)

print("Running dry-run calculation...")
print()

results = run_locally(job, create_folders=True)
