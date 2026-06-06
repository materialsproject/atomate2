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
# Method 1: Forward Construction (XC → Path)
# ============================================================================
print("\n### Method 1: Forward Construction (Recommended for new calculations)")
print("Provide: XC functional → System constructs path")

maker_forward = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    user_params={
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [6, 6, 6],
        "xc.functional": "GGA",
        "xc.authors": "PBEsol",
        "a2s_pseudo_relativistic": "SR",
    },
)

job_forward = maker_forward.make(structure)
run_locally(job_forward, create_folders=True)

print("\nInput:")
print("  xc.functional: GGA")
print("  xc.authors: PBEsol")
print("  pseudo_relativistic: SR")
print("\nOutput:")
print("  Constructed path: .../ONCVPSP-PBEsol-SR-PDv0.4-Standard")
print("  ✓ Path automatically constructed from XC parameters")
