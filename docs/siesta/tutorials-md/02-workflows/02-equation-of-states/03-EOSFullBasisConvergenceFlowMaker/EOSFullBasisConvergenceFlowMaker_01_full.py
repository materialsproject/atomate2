#!/usr/bin/env python
"""
Full Basis Parameter Optimization with EOS.

This example demonstrates EOSFullBasisConvergenceMaker, which tests ALL combinations
of basis sizes, PAO.EnergyShift, and PAO.SplitNorm to find optimal parameters.

Comparison with EOSBasisConvergenceMaker:
- EOSBasisConvergenceMaker (01-03.py): Tests only basis sets with FIXED parameters
  Example: DZ (ES=0.01), DZP (ES=0.01), TZP (ES=0.005) → 3 EOS calculations

- EOSFullBasisConvergenceMaker (this file): Tests ALL parameter combinations
  Example: 2 basis × 3 ES × 2 SN = 12 EOS calculations

Use this when you need to optimize PAO parameters for each basis set.
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.eos import EOSFullBasisConvergenceFlowMaker

# Load structure
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Create full parameter optimization workflow
# This tests: 2 basis × 3 energy_shifts × 2 split_norms = 12 EOS calculations
# Each EOS has 5 volume points, so total = 12 × 5 = 60 SIESTA runs
maker = EOSFullBasisConvergenceFlowMaker(
    dry_run=False,
    basis_sizes=["DZ", "DZP"],  # Test 2 basis sizes
    energy_shifts=[0.01, 0.015],  # , 0.02],  # Test 3 PAO.EnergyShift values (Ry)
    split_norms=[0.15, 0.20],  # Test 2 PAO.SplitNorm values
    linear_strain=(-0.04, 0.04),  # ±4% volume strain
    number_of_frames=5,  # 5 volume points per EOS
    a2s_kpts=[4, 4, 4],  # K-point grid
)

# Run the workflow
workflow = maker.make(structure)
results = run_locally(workflow, create_folders=True, root_dir="02_full")

print("✓ Full basis parameter optimization complete!")
print("  - Tested 2 basis × 3 energy shifts × 2 split norms = 12 combinations")
print("  - Each combination has its own EOS fit")
print("  - Check output for optimal parameters for each basis")
