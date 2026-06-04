#!/usr/bin/env python
"""
Tutorial 11: PAO.Basis Helper Functions (Custom Orbital Specifications)

This tutorial demonstrates the PAO.Basis helper functions for creating
custom orbital specifications programmatically.

Use Case:
---------
When you need full control over orbital parameters:
- Custom cutoff radii for each zeta function
- Polarization orbitals with specific parameters
- Split norm values, soft confinement, charge confinement
- Species variants with different orbital configurations

Key Feature (NEW in v1.0.0):
----------------------------
Dataclass-based system for building %block PAO.Basis programmatically,
with validation and support for all SIESTA PAO flags.

Author: Arsalan Akhtar
Date: 2025-12-14
"""

from pathlib import Path

from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.utils.basis_builder import create_pao_basis

# Create output directory
output_dir = Path("dry_run_output_11_pao_basis_helpers")
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("Tutorial 11: PAO.Basis Helper Functions")
print("=" * 70)

# Create TiO2 structure
lattice = Lattice.from_parameters(4.6, 4.6, 3.0, 90, 90, 90)
structure = Structure(
    lattice,
    ["Ti", "Ti", "O", "O", "O", "O"],
    [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
        [0.3, 0.3, 0],
        [0.7, 0.7, 0],
        [0.2, 0.8, 0.5],
        [0.8, 0.2, 0.5],
    ],
)

print(f"\n1️⃣  Structure: {structure.composition}")
print(f"   Lattice: {structure.lattice.abc}")

# ============================================================================
# Example 1: Simple Custom Basis (Basic)
# ============================================================================
print("\n" + "=" * 70)
print("Example 1: Simple Custom Basis")
print("=" * 70)

print("\n📋 Define custom basis for Ti and O:")

basis_spec_simple = {
    "Ti": {
        "shells": [
            {"n": 4, "l": 0, "nzeta": 2, "rc": [6.0, 0.0]},  # 4s: 2 zeta
            {"n": 3, "l": 2, "nzeta": 2, "rc": [7.0, 0.0]},  # 3d: 2 zeta
        ]
    },
    "O": {
        "shells": [
            {"n": 2, "l": 0, "nzeta": 2, "rc": [5.0, 0.0]},  # 2s: 2 zeta
            {"n": 2, "l": 1, "nzeta": 2, "rc": [6.0, 0.0]},  # 2p: 2 zeta
        ]
    },
}

print("\n   basis_spec = {")
print("       'Ti': {")
print("           'shells': [")
print("               {'n': 4, 'l': 0, 'nzeta': 2, 'rc': [6.0, 0.0]},  # 4s")
print("               {'n': 3, 'l': 2, 'nzeta': 2, 'rc': [7.0, 0.0]},  # 3d")
print("           ]")
print("       },")
print("       'O': { ... }")
print("   }")

# Generate PAO.Basis block
pao_basis_simple = create_pao_basis(basis_spec_simple)

print("\n✅ Generated %block PAO.Basis:")
print("\n   %block PAO.Basis")
for line in pao_basis_simple:
    print(f"   {line}")
print("   %endblock PAO.Basis")

# ============================================================================
# Example 2: Custom Basis with Polarization
# ============================================================================
print("\n" + "=" * 70)
print("Example 2: Custom Basis with Polarization Orbitals")
print("=" * 70)

print("\n📋 Add polarization orbitals for higher accuracy:")

basis_spec_pol = {
    "Ti": {
        "shells": [
            {"n": 4, "l": 0, "nzeta": 2, "rc": [6.0, 0.0]},  # 4s: 2 zeta
            {
                "n": 3,
                "l": 2,
                "nzeta": 2,
                "rc": [7.0, 0.0],
                "polarization": True,
            },  # 3d: 2 zeta + pol
        ]
    },
    "O": {
        "shells": [
            {"n": 2, "l": 0, "nzeta": 2, "rc": [5.0, 0.0]},  # 2s: 2 zeta
            {
                "n": 2,
                "l": 1,
                "nzeta": 2,
                "rc": [6.0, 0.0],
                "polarization": True,
            },  # 2p: 2 zeta + pol
        ]
    },
}

print("\n   Key addition: 'polarization': True")
print("      • Adds l+1 orbital (e.g., d orbital for p shell)")
print("      • Improves description of bonding, charge transfer")

pao_basis_pol = create_pao_basis(basis_spec_pol)

print("\n✅ Generated %block PAO.Basis (with polarization):")
print("\n   %block PAO.Basis")
for line in pao_basis_pol:
    print(f"   {line}")
print("   %endblock PAO.Basis")

# ============================================================================
# Example 3: Species Variants with Different Orbitals
# ============================================================================
print("\n" + "=" * 70)
print("Example 3: Species Variants (Surface vs Bulk)")
print("=" * 70)

print("\n📋 Different orbital configurations for surface vs bulk:")

basis_spec_variants = {
    "O_surface": {
        "shells": [
            {"n": 2, "l": 0, "nzeta": 2, "rc": [6.0, 0.0]},  # Longer rc (diffuse)
            {
                "n": 2,
                "l": 1,
                "nzeta": 2,
                "rc": [7.0, 0.0],
                "polarization": True,
            },
        ]
    },
    "O_bulk": {
        "shells": [
            {"n": 2, "l": 0, "nzeta": 2, "rc": [4.5, 0.0]},  # Shorter rc (compact)
            {"n": 2, "l": 1, "nzeta": 2, "rc": [5.5, 0.0]},  # No polarization
        ]
    },
}

print("\n   Surface O: Longer cutoff radii (more diffuse, better for bonding)")
print("   Bulk O: Shorter cutoff radii (compact, efficient)")

pao_basis_variants = create_pao_basis(basis_spec_variants)

print("\n✅ Generated %block PAO.Basis (species variants):")
print("\n   %block PAO.Basis")
for line in pao_basis_variants:
    print(f"   {line}")
print("   %endblock PAO.Basis")

# ============================================================================
# Example 4: Advanced - All PAO Flags
# ============================================================================
print("\n" + "=" * 70)
print("Example 4: Advanced PAO Flags (Expert Level)")
print("=" * 70)

print("\n📋 Full control with all SIESTA PAO flags:")

basis_spec_advanced = {
    "O": {
        "shells": [
            {
                "n": 2,
                "l": 1,
                "nzeta": 2,
                "rc": [6.0, 0.0],
                "split_norm": 0.25,  # Split norm for second zeta
                "charge_conf": 0.5,  # Charge confinement
                "soft_conf": True,  # Soft confinement potential
                "polarization": True,
            }
        ]
    }
}

print("\n   Available PAO flags:")
print("      • split_norm: Controls second zeta generation (0.0-1.0)")
print("      • charge_conf: Charge confinement for excited states")
print("      • soft_conf: Soft confinement potential (vs hard)")
print("      • filteret: Filtering threshold")
print("      • screen: Screening of orbitals")
print("      • delta: Delta for numerical derivatives")
print("      • contraction: Orbital contraction")

pao_basis_advanced = create_pao_basis(basis_spec_advanced)

print("\n✅ Generated %block PAO.Basis (all flags):")
print("\n   %block PAO.Basis")
for line in pao_basis_advanced:
    print(f"   {line}")
print("   %endblock PAO.Basis")

# ============================================================================
# Integration with RelaxMaker
# ============================================================================
print("\n" + "=" * 70)
print("Integration with RelaxMaker")
print("=" * 70)

print("\n2️⃣  Using custom PAO.Basis in workflow:")

# Use the polarization example
user_params = {
    "%block PAO.Basis": pao_basis_pol,  # Helper function returns list format!
    "PAO.BasisSize": "DZP",  # Fallback for species not in PAO.Basis
    "Mesh.Cutoff": "300 Ry",
    "a2s_kpts": [4, 4, 4],
}

print("\n   user_params = {")
print("       '%block PAO.Basis': pao_basis_pol,  # Already in list format!")
print("       'PAO.BasisSize': 'DZP',  # Fallback")
print("       'Mesh.Cutoff': '300 Ry',")
print("       'a2s_kpts': [4, 4, 4],")
print("   }")

maker = RelaxMaker.fixed_cell_relaxation(user_params=user_params, dry_run=True)

job = maker.make(structure)
response = run_locally(job, create_folders=True, root_dir=str(output_dir))

print("\n3️⃣  Calculation completed (dry-run mode)")
print(f"   📁 Output: {output_dir}/job_*/siesta.fdf")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary: PAO.Basis Helper Functions")
print("=" * 70)

print("\n✅ Advantages:")
print("   • Programmatic generation (no manual FDF string formatting)")
print("   • Validation: checks nzeta vs rc length, l range, split_norm")
print("   • Type safety: Dataclass-based with proper types")
print("   • All SIESTA PAO flags supported")
print("   • Species variants: O_surface, O_bulk, O_ghost, etc.")
print("   • Returns list format (ready for user_params)")

print("\n🎯 Use Cases:")
print("   • Custom cutoff radii optimization")
print("   • Species variants with different orbitals")
print("   • Polarization orbital control")
print("   • Soft confinement, charge confinement")
print("   • Advanced basis set development")

print("\n💡 Code Pattern:")
print(
    """
   from atomate2.siesta.sets.utils.basis_builder import create_pao_basis

   # Define basis specification
   basis_spec = {
       'O': {
           'shells': [
               {'n': 2, 'l': 1, 'nzeta': 2, 'rc': [6.0, 0.0],
                'polarization': True, 'split_norm': 0.25}
           ]
       }
   }

   # Generate PAO.Basis block
   pao_basis = create_pao_basis(basis_spec)

   # Use in RelaxMaker
   maker = RelaxMaker(user_params={'%block PAO.Basis': pao_basis})
"""
)

print("\n📚 Related Helper Functions:")
print("   • PAOShell: Dataclass for single orbital shell")
print("   • PAOBasisSpecies: Dataclass for complete species basis")
print("   • create_pao_basis(): High-level dict → FDF conversion")

print("\n⚠️  Important:")
print("   • Helper returns list format (ready to use)")
print("   • Coexists with PAO.BasisSize (fallback for other species)")
print("   • PAO.Basis takes priority over PAO.BasisSizes for its species")
print("   • Validation errors raise ValueError with helpful messages")

print("\n" + "=" * 70)
print("Tutorial Complete! ✨")
print("=" * 70)
print(f"\n📁 Check FDF file: {output_dir}/job_*/siesta.fdf")
print("   Look for:")
print("      • %block PAO.Basis (custom orbital specifications)")
print("      • PAO.BasisSize (fallback for Ti)")
print("\n📚 Related:")
print("   • Tutorial 04-07: Manual PAO.Basis blocks")
print("   • Tutorial 08-10: Species variants")
print("   • Tutorial 03: Per-atom basis helpers")
print("\n💡 Pro Tip:")
print("   Combine with per-atom helpers for ultimate control:")
print("      1. Use apply_per_atom_basis() for species labels")
print("      2. Use create_pao_basis() for custom orbitals per species")
print("      3. Enjoy atom-level precision with custom orbital control!")
