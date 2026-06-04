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
# Method 3: Explicit XC Override
# ============================================================================
print("\n" + "=" * 80)
print("### Method 3: Explicit Override (Advanced use case)")
print("Provide: Both path AND XC → Explicit XC overrides auto-detection")

maker_override = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    user_params={
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [6, 6, 6],
        "a2s_pseudo_path": "/Users/aakhtar/.siesta/pseudos/ONCVPSP-PBE-FR-PDv0.4-Standard",
        # Explicitly override the PBE from path
        "xc.functional": "GGA",
        "xc.authors": "PW91",
    },
)

job_override = maker_override.make(structure)
run_locally(job_override, create_folders=True)

print("\nInput:")
print("  pseudo_path: .../ONCVPSP-PBE-FR-PDv0.4-Standard")
print("  xc.authors: PW91 (explicit override)")
print("\nOutput:")
print("  XC.Authors = PW91 (explicit value used, not PBE from path)")
print("  ⚠️  Warning: This may cause XC mismatch!")
print("  → Explicit XC overrides auto-detection")
