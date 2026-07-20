#!/usr/bin/env python
"""O(N) domain and spatial decomposition for large-scale calculations."""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

# Create Si supercell
si_primitive = Structure.from_spacegroup(
    "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]
)
structure = si_primitive.copy()
structure.make_supercell([2, 2, 1])
print(f"Structure: {structure.composition.reduced_formula}, {len(structure)} atoms")

# Configure O(N) decomposition
user_params = {
    "UseDomainDecomposition": True,  # Group orbitals/atoms by processor
    "UseSpatialDecomposition": True,  # Distribute real-space grid
    "RcSpatial": 15.0,  # 15 Bohr communication radius
    "SolutionMethod": "OMM",  # Order-N Minimization Method
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [2, 2, 2],
    "Mesh.Cutoff": "150 Ry",
    "MD.TypeOfRun": "CG",
    "MD.NumCGsteps": 0,  # Single point for demo
}
print(
    f"O(N) config: DomainDecomp={user_params['UseDomainDecomposition']}, RcSpatial={user_params['RcSpatial']} Bohr"
)

# Create and run O(N) job (requires tier="expert" for ParallelOptions)
maker = RelaxMaker.fixed_cell_relaxation(
    user_params=user_params, tier="expert", dry_run=False
)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ Generated siesta.fdf with O(N) parallel configuration")
print("\nKey points:")
print("  - Domain decomposition: Groups orbitals/atoms (requires O(N) solver)")
print("  - Spatial decomposition: Distributes real-space grid")
print("  - RcSpatial: 10-30 Bohr typical (smaller = less comm, less accurate)")
print("  - Use for: >1000 atoms with OMM/PEXSI solvers")
