#!/usr/bin/env python
"""Band structure calculation without relaxation.

Use this approach when:
- You have a pre-relaxed structure
- You want faster calculations
- You're testing parameters before full production runs

The workflow still performs:
1. SCF calculation for ground state
2. Band structure along high-symmetry k-path
3. Analysis and plotting

Runtime: ~15-20 minutes (no relaxation step)
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.bands import BandStructureFlowMaker

# Load silicon primitive cell (assume already relaxed)
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Create band structure workflow WITHOUT relaxation
# Set relax_maker=None to skip relaxation
maker = BandStructureFlowMaker(
    relax_maker=None,  # Skip relaxation
    plot_bands=True,  # Still generate plot
)

# Generate and run workflow
flow = maker.make(structure)
results = run_locally(flow, create_folders=True, root_dir="02_skip_relaxation")

print("\n" + "=" * 60)
print("Band Structure (No Relaxation) Complete!")
print("=" * 60)
print("\nThis workflow skipped relaxation for faster results.")
print("Use this for pre-relaxed structures or quick tests.")
