#!/usr/bin/env python
"""Dry run mode for band structure workflow.

Dry run mode is useful for:
- Previewing workflow structure before running
- Validating input parameters
- Generating SIESTA input files without running calculations
- Testing on systems without SIESTA installed

In dry run mode:
- Input files are generated but SIESTA is NOT executed
- Saves ~99% of computational time
- Perfect for debugging and parameter testing

Runtime: ~1-2 seconds
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.bands import BandStructureFlowMaker

# Load silicon primitive cell
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Create band structure workflow with dry_run enabled
# All child makers automatically inherit dry_run=True
maker = BandStructureFlowMaker(
    dry_run=True,  # Enable dry run mode
    dry_run_output_dir="dry_run_output",  # Where to save input files
    dry_run_format="cif",  # Structure output format
)

# Generate and run workflow (will NOT execute SIESTA)
flow = maker.make(structure)
results = run_locally(flow, create_folders=True, root_dir="05_dry_run")

print("\n" + "=" * 60)
print("Dry Run Complete!")
print("=" * 60)
print("\nNo SIESTA calculations were executed.")
print("Check 'dry_run_output/' for generated input files:")
print("  - siesta.fdf: SIESTA input file")
print("  - structure.cif: Input structure")
print("\nUse dry run to validate parameters before production runs.")
