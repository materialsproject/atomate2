#!/usr/bin/env python
"""Pt(111) surface energy with diffuse basis for surface atoms.

This tutorial demonstrates MultiSurfaceEnergyFlowMaker with diffuse basis sets
for surface atoms. Surface atoms need larger (more diffuse) basis sets because
electrons extend further into the vacuum than in bulk.

Diffuse basis benefits:
- Improved surface energies
- Better work functions
- More accurate adsorption energies

The apply_diffuse_basis parameter automatically:
1. Detects surface atoms (outermost atomic layers)
2. Assigns larger basis (DZP) to surface atoms
3. Assigns standard basis (DZ) to bulk/interior atoms
4. Generates the PAO.BasisSizes block for SIESTA

Output files:
- multi_surface_summary.txt: Surface energy results
- pt_slab_energy.json: Slab energy for reuse in adsorption calculations
"""

import json
from pathlib import Path

from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.siesta.flows.surface import MultiSurfaceEnergyFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

# ============================================================================
# Create Pt bulk structure (FCC)
# ============================================================================
a = 3.92  # Pt lattice constant (Angstrom)

# Create FCC Pt using space group symmetry (Fm-3m, #225)
pt_bulk = Structure.from_spacegroup(
    sg="Fm-3m",
    lattice=Lattice.cubic(a),
    species=["Pt"],
    coords=[[0, 0, 0]],
)

print("Pt bulk structure (FCC):")
print("  Space group: Fm-3m")
print(f"  Lattice constant: {a} A")
print(f"  Atoms per unit cell: {len(pt_bulk)}")

# ============================================================================
# Create makers with tier presets
# ============================================================================
bulk_maker = StaticMaker(use_custodian=True, custodian_max_errors=10)
bulk_maker = apply_tier_preset(bulk_maker, "surface_metal")

slab_maker = StaticMaker(use_custodian=True, custodian_max_errors=10)
slab_maker = apply_tier_preset(
    slab_maker,
    "surface_metal",
    # override_params={"a2s_kpts": [4, 4, 1]},  # Reduced k-points in vacuum direction
)

# ============================================================================
# Create MultiSurfaceEnergyFlowMaker with diffuse basis
# ============================================================================
maker = MultiSurfaceEnergyFlowMaker(
    name="Pt_surface_energy",
    miller_indices=[(1, 1, 1)],  # Calculate Pt(111) surface
    bulk_static_maker=bulk_maker,
    slab_static_maker=slab_maker,
    slab_layers=7,  # 7 atomic layers (more layers for better convergence)
    vacuum_size=15.0,  # 15 A vacuum
    symmetrize=True,  # Request symmetric slab
    # Diffuse basis parameters
    apply_diffuse_basis=True,  # Enable diffuse basis for surface atoms
    surface_basis="DZP",  # Larger basis for surface atoms
    bulk_basis="DZ",  # Standard basis for interior atoms
    surface_layers=1,  # 1 outermost layer per surface (top + bottom)
    dry_run=True,
    # Note: termination parameter not needed for single-element metals (Pt)
    # All surfaces are Pt-terminated by definition
    #
    # IMPORTANT: FCC (111) slabs from pymatgen may have small x,y asymmetry
    # (top and bottom surfaces at different stacking positions) due to ABC stacking.
    # For pure metals like Pt, this is usually acceptable because there's no
    # charge separation (unlike polar compounds like MgO).
    # The code will warn if truly symmetric slab cannot be found.
)

flow = maker.make(pt_bulk)

# ============================================================================
# Run workflow
# ============================================================================
output_dir = Path("05_pt111_diffuse_basis")
output_dir.mkdir(exist_ok=True)

print("\nRunning Pt(111) surface energy calculation with diffuse basis...")
print("  Surface atoms will use DZP basis (more diffuse)")
print("  Bulk atoms will use DZ basis (standard)")

responses = run_locally(
    flow,
    create_folders=True,
    root_dir=str(output_dir),
    ensure_success=True,
)

# ============================================================================
# Extract and save results
# ============================================================================
print("\n" + "=" * 60)
print("Results")
print("=" * 60)

for uuid, response in responses.items():
    if hasattr(response, "output") and response.output is not None:
        output = response.output
        if isinstance(output, dict) and "all_results" in output:
            for result in output["all_results"]:
                hkl = result["miller_index"]
                for term in result["terminations"]:
                    print(f"\nPt({hkl[0]}{hkl[1]}{hkl[2]}) surface:")
                    print(f"  Surface energy: {term['surface_energy']:.6f} eV/A^2")
                    print(f"  Surface energy: {term['surface_energy_Jm2']:.3f} J/m^2")
                    print(f"  Slab energy: {term['slab_energy']:.6f} eV")
                    print(f"  N atoms: {term['n_atoms']}")

                    # Save slab energy for reuse in adsorption calculations
                    slab_data = {
                        "slab_energy": term["slab_energy"],
                        "surface_energy_eV_A2": term["surface_energy"],
                        "surface_energy_Jm2": term["surface_energy_Jm2"],
                        "miller_index": list(hkl),
                        "n_atoms": term["n_atoms"],
                        "surface_area": term["surface_area"],
                        "apply_diffuse_basis": True,
                        "surface_basis": "DZP",
                        "bulk_basis": "DZ",
                    }
                    with open(output_dir / "pt111_slab_energy.json", "w") as f:
                        json.dump(slab_data, f, indent=2)
                    print(
                        f"\nSaved slab energy to {output_dir / 'pt111_slab_energy.json'}"
                    )

print("\n" + "=" * 60)
print("To reuse slab energy in adsorption calculations:")
print("=" * 60)
print(
    """
import json
from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker

with open("05_pt111_diffuse_basis/pt111_slab_energy.json") as f:
    data = json.load(f)

maker = AdsorptionScanFlowMaker(
    precalc_slab_energy=data["slab_energy"],
    ...
)
"""
)

print("\n✓ Pt(111) surface energy calculation with diffuse basis complete")
