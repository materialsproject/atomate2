#!/usr/bin/env python
"""Combined mesh cutoff and k-points convergence - Basic Example.

This tutorial demonstrates the MeshKpointConvergenceFlowMaker for combined
two-stage convergence testing with default energy-only criteria.
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.convergence import MeshKpointConvergenceFlowMaker

# Load structure
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

print("=" * 80)
print("Combined Mesh Cutoff + K-points Convergence (Basic)")
print("=" * 80)
print()

# Create combined convergence workflow
# Default: Only energy convergence (1 meV tolerance)
maker = MeshKpointConvergenceFlowMaker(
    mesh_cutoffs=[200, 250, 300, 350, 400],
    kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8]],
    stage1_kpoints=[1, 1, 1],  # Coarse k-points for mesh convergence
    dry_run=False,
)

print("Workflow Configuration:")
print("  Stage 1: Mesh cutoff convergence")
print(f"    - Mesh cutoffs: {maker.mesh_cutoffs}")
print(f"    - K-points (fixed): {maker.stage1_kpoints}")
print("  Stage 2: K-points convergence")
print(f"    - K-points: {maker.kpoints_list}")
print(f"    - Mesh cutoff: {maker.mesh_cutoffs[-1]} Ry (from Stage 1)")
print()
print("  Convergence criterion: ΔE < 1.0 meV")
print()

# Create workflow
flow = maker.make(structure)

print(f"✓ Created workflow with {len(flow.jobs)} jobs:")
print(f"  - {len(maker.mesh_cutoffs)} mesh cutoff tests (Stage 1)")
print(f"  - {len(maker.kpoints_list)} k-point tests (Stage 2)")
print("  - 4 collection/plotting jobs")
print()

# Run workflow
results = run_locally(flow, create_folders=True)

print("=" * 80)
print("Workflow Complete!")
print("=" * 80)
print()
print("Output files generated:")
print()
print("Stage 1 (Mesh Cutoff):")
print("  - convergence_mesh_cutoff_energy.png")
print("  - convergence_mesh_cutoff_convergence.png")
print("  - convergence_mesh_cutoff_fermi.png")
print("  - convergence_mesh_cutoff.txt")
print()
print("Stage 2 (K-points):")
print("  - convergence_kpoints_energy.png")
print("  - convergence_kpoints_convergence.png")
print("  - convergence_kpoints_fermi.png")
print("  - convergence_kpoints.txt")
print()
print("✓ Check the .txt files for detailed convergence analysis!")
