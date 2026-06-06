#!/usr/bin/env python
"""
Full Basis Parameter Optimization with EOS and Custodian Error Handling.

This example demonstrates EOSFullBasisConvergenceMaker with automatic error recovery
using Custodian. This is particularly useful when testing many parameter combinations
where some may have SCF convergence issues.

Custodian Features:
- Automatic SCF convergence rescue (5-level progressive strategies)
- Mixer weight adjustment, occupation function changes, DM reinitialization
- high error recovery rate even with difficult parameter combinations
- Automatic propagation to all child makers (initial_relax_maker, eos_relax_maker)

How custodian propagation works:
- EOSFullBasisConvergenceFlowMaker inherits from BaseSiestaFlowMaker
- Setting use_custodian=True triggers __post_init__() propagation
- Both initial_relax_maker and eos_relax_maker automatically receive:
  - use_custodian=True
  - custodian_max_errors (from flow maker)
  - custodian_handlers (if provided)

When to use custodian with EOSFullBasisConvergenceFlowMaker:
- Testing aggressive PAO.EnergyShift values (< 0.005 Ry may have convergence issues)
- Testing extreme PAO.SplitNorm values (< 0.10 or > 0.30)
- Running large parameter sweeps where occasional failures would waste compute time
- Production runs on HPC clusters where job failures are costly
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.eos import EOSFullBasisConvergenceFlowMaker

# Load structure
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Create full parameter optimization workflow WITH CUSTODIAN
# Testing: 2 basis × 2 energy_shifts × 2 split_norms = 8 EOS calculations
# Each EOS has 5 volume points, so total = 8 × 5 = 40 SIESTA runs
# Custodian automatically handles any SCF convergence issues
maker = EOSFullBasisConvergenceFlowMaker(
    dry_run=False,
    # Basis parameters to test
    basis_sizes=["DZ", "DZP"],  # Test 2 basis sizes
    energy_shifts=[0.01, 0.015],  # Test 2 PAO.EnergyShift values (Ry)
    split_norms=[0.15, 0.20],  # Test 2 PAO.SplitNorm values
    # EOS settings
    linear_strain=(-0.04, 0.04),  # ±4% volume strain
    number_of_frames=5,  # 5 volume points per EOS
    a2s_kpts=[4, 4, 4],  # K-point grid
    # Custodian error handling (propagates to ALL child makers automatically)
    use_custodian=True,  # Enable automatic error recovery
    custodian_max_errors=10,  # Allow up to 10 recovery attempts per job
)

# Run the workflow
workflow = maker.make(structure)
results = run_locally(workflow, create_folders=True, root_dir="02_with_custodian")

print("✓ Full basis parameter optimization with custodian complete!")
print("  - Tested 2 basis × 2 energy shifts × 2 split norms = 8 combinations")
print("  - Each combination has its own EOS fit (5 volume points)")
print("  - Custodian automatically handled any SCF convergence issues")
print("  - Check output for optimal parameters for each basis")
