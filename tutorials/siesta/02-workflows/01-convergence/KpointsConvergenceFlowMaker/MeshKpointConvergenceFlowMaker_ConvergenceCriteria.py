#!/usr/bin/env python
"""Combined convergence with multiple criteria.

This tutorial demonstrates using multiple convergence criteria to ensure
all important properties are converged before stopping tests.
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.convergence import (
    ConvergenceCriteria,
    MeshKpointConvergenceFlowMaker,
)

# Load structure
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

print("=" * 80)
print("Multi-Criteria Convergence")
print("=" * 80)
print()

# Define strict convergence criteria for all properties
criteria = ConvergenceCriteria(
    energy_tol=1.0,  # 1 meV energy difference
    fermi_tol=0.01,  # 0.01 eV Fermi energy difference
    force_tol=0.01,  # 0.01 eV/Å maximum force
    stress_tol=0.05,  # 0.05 eV/Å³ maximum stress
)

print("Convergence Criteria:")
print(f"  {criteria}")
print()
print("  ALL criteria must be satisfied to declare convergence")
print("  Tests will stop when 2 consecutive points are converged")
print()

# Create workflow with multi-criteria convergence
maker = MeshKpointConvergenceFlowMaker(
    mesh_cutoffs=[150, 200, 250, 300, 350, 400, 450],
    kpoints_list=[[2, 2, 2], [3, 3, 3], [4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10]],
    stage1_kpoints=[4, 4, 4],
    convergence_criteria=criteria,
    require_consecutive=2,
    dry_run=False,
)

flow = maker.make(structure)

print("✓ Workflow created with up to:")
print(f"  - {len(maker.mesh_cutoffs)} mesh cutoff tests (Stage 1)")
print(f"  - {len(maker.kpoints_list)} k-point tests (Stage 2)")
print()
print("  Workflow will stop early if all criteria are met!")
print()

# Run workflow
results = run_locally(flow, create_folders=True)

print("=" * 80)
print("Workflow Complete!")
print("=" * 80)
print()
print("Check convergence_*.txt files for:")
print("  ✓ Energy convergence analysis")
print("  ✓ Fermi energy statistics")
print("  ✓ Force convergence assessment")
print("  ✓ Stress statistics")
print()
print("All plots generated with comprehensive property tracking!")
