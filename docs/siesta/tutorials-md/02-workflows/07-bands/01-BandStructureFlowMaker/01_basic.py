#!/usr/bin/env python
"""Basic band structure workflow with automatic relaxation.

This tutorial demonstrates the complete BandStructureFlowMaker workflow:
1. Structure relaxation (variable cell)
2. SCF calculation for ground state
3. Band structure along high-symmetry k-path
4. Analysis and plotting

The workflow automatically:
- Generates the appropriate k-path for the crystal symmetry
- Extracts band gap, VBM, CBM information
- Creates publication-quality band structure plot
- Writes a summary file with electronic properties

Runtime: ~30 minutes (depending on system size and parameters)
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.bands import BandStructureFlowMaker

# Load silicon primitive cell
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")
# structure = Structure.from_file("Si.cif")

# Create band structure workflow with default settings
# This includes:
# - Variable-cell relaxation
# - SCF calculation with DZP basis, 300 Ry cutoff
# - Band structure along automatic k-path
# - Band gap analysis and plotting
maker = BandStructureFlowMaker()

# Generate and run workflow
flow = maker.make(structure)
results = run_locally(flow, create_folders=True, root_dir="01_basic")

print("\n" + "=" * 60)
print("Band Structure Workflow Complete!")
print("=" * 60)
print("\nOutput files:")
print("  - band_structure_summary.txt: Electronic properties")
print("  - band_structure.png: Band structure plot")
print("\nCheck the summary file for band gap information.")
