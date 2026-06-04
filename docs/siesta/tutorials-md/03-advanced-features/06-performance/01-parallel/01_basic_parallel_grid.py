#!/usr/bin/env python
"""Basic parallel processor grid configuration for matrix distribution."""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

# Create Si structure
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
print(f"Structure: {structure.composition.reduced_formula}, {len(structure)} atoms")

# Configure parallel processor grid
user_params = {
    "BlockSize": 32,  # 32x32 blocks for matrix distribution
    "ProcessorY": 4,  # 4 processors in Y dimension (16 procs = 4x4 grid)
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [4, 4, 4],
    "Mesh.Cutoff": "200 Ry",
}
print(
    f"Parallel config: BlockSize={user_params['BlockSize']}, ProcessorY={user_params['ProcessorY']}"
)

# Create and run job (requires tier="expert" for ParallelOptions)
maker = StaticMaker.scf(user_params=user_params, tier="expert", dry_run=False)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ Generated siesta.fdf with parallel configuration")
print("\nKey points:")
print("  - BlockSize: 16-64 typical (smaller = better load balance, more comm)")
print("  - ProcessorY: Sets 2D grid (square grids usually best)")
print("  - Default: SIESTA auto-selects (start here, tune if needed)")
