#!/usr/bin/env python
"""Example 5: With Convergence Testing - Automatically use converged parameters.

NEW FEATURE (v1.0.0):
When test_convergence=True, the workflow now:
1. Runs convergence tests for k-points and mesh cutoff
2. Automatically extracts optimal converged parameters
3. Uses these converged parameters for ALL subsequent property calculations

This ensures that all electronic/mechanical/thermal properties are calculated
with properly converged computational parameters!
"""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create example structure (Silicon)
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Complete study with automatic convergence-based parameter selection
flow = RecipeBook.complete_material_study(
    silicon,
    test_convergence=True,  # Enable convergence testing
    # NEW: Converged k-points and mesh cutoff will be AUTOMATICALLY used
    # for all property calculations (electronic, mechanical, thermal)!
)

# What happens under the hood:
# 1. Convergence suite runs and finds optimal k-points (e.g., [6,6,6])
# 2. Convergence suite finds optimal mesh cutoff (e.g., "350 Ry")
# 3. Extract job pulls these optimal values from convergence results
# 4. Merge job combines converged params with any user_params you specified
# 5. Property workflows (electronic/mechanical/thermal) use the converged params!

# Run locally
results = run_locally(flow, create_folders=True)

# You can also override specific parameters while using converged ones:
# flow = RecipeBook.complete_material_study(
#     silicon,
#     test_convergence=True,  # Find optimal k-points and cutoff
#     user_params={
#         "PAO.BasisSize": "TZP",  # Override basis size
#         # Converged k-points and cutoff will still be used!
#     }
# )
