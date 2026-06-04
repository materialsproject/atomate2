#!/usr/bin/env python
"""
Automatic XC Functional Detection from Pseudopotential Path.

This tutorial demonstrates the bidirectional pseudopotential system:
1. Forward: Provide XC functional → System constructs pseudopotential path
2. Reverse: Provide pseudopotential path → System detects XC functional

The reverse detection is useful when you already know which pseudopotential
directory to use and want the system to automatically set the correct XC functional.
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.jobs.core import RelaxMaker

# Load structure
structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")


# ============================================================================
# Method 2: Reverse Detection (Path → XC)
# ============================================================================
print("\n" + "=" * 80)
print("### Method 2: Reverse Detection (Convenient when path is known)")
print("Provide: Full pseudopotential path → System detects XC")

maker_reverse = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    user_params={
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [6, 6, 6],
        # Only provide the path - XC will be auto-detected!
        "a2s_pseudo_path": "/Users/aakhtar/.siesta/pseudos/ONCVPSP-PBE-FR-PDv0.4-Standard",
    },
)

job_reverse = maker_reverse.make(structure)
run_locally(job_reverse, create_folders=True)

print("\nInput:")
print("  pseudo_path: /Users/aakhtar/.siesta/pseudos/ONCVPSP-PBE-FR-PDv0.4-Standard")
print("  (No XC parameters provided!)")
print("\nOutput:")
print("  Auto-detected: XC.Functional = GGA, XC.Authors = PBE")
print("  → System parsed 'PBE' from directory name")
print("  ✓ XC functional automatically detected from path")
