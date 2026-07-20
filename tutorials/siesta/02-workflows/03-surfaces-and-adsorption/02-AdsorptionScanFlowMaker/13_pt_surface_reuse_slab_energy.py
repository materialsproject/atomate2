#!/usr/bin/env python
"""Reuse slab energy for additional adsorbates on Pt(111).

This tutorial demonstrates how to use a pre-calculated slab energy to scan
additional adsorbates without recalculating the clean slab.

PREREQUISITE: Run one of these first to generate the slab energy:
  Option A: 12_pt_surface_first_adsorbate.py (adsorption scan, saves slab energy)
  Option B: ../01-MultiSurfaceEnergyFlowMaker/05_pt111_diffuse_basis.py (slab only)

Expected files from Option A (12_pt_surface_first_adsorbate.py):
- 12_pt_surface_first_adsorbate/pt111_energies.json (contains slab_energy)
- 12_pt_surface_first_adsorbate/pt111_slab.cif (the slab structure)
- 12_pt_surface_first_adsorbate/pao_basissizes.txt (basis set configuration)

Expected files from Option B (05_pt111_diffuse_basis.py):
- ../01-MultiSurfaceEnergyFlowMaker/05_pt111_diffuse_basis/pt111_slab_energy.json
- ../01-MultiSurfaceEnergyFlowMaker/05_pt111_diffuse_basis/pt111_slab.cif
- ../01-MultiSurfaceEnergyFlowMaker/05_pt111_diffuse_basis/pao_basissizes.txt

Performance benefit:
- Skips the clean slab DFT calculation entirely
- For a 12-atom Pt slab, this saves significant compute time
- Essential for screening multiple adsorbates on the same surface
"""

import json
from pathlib import Path

from jobflow import run_locally
from pymatgen.core import Molecule, Structure

from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

# ============================================================================
# Load pre-calculated data from previous tutorial
# ============================================================================
# Try Option A first (adsorption tutorial), then Option B (slab-only tutorial)
option_a_dir = Path("12_pt_surface_first_adsorbate")
option_b_dir = Path("../01-MultiSurfaceEnergyFlowMaker/05_pt111_diffuse_basis")

if option_a_dir.exists():
    source_dir = option_a_dir
    energies_filename = "pt111_energies.json"
    print(f"Using data from Option A: {source_dir}")
elif option_b_dir.exists():
    source_dir = option_b_dir
    energies_filename = "pt111_slab_energy.json"
    print(f"Using data from Option B: {source_dir}")
else:
    raise FileNotFoundError(
        "No pre-calculated data found.\n"
        "Please run one of these first:\n"
        "  Option A: 12_pt_surface_first_adsorbate.py\n"
        "  Option B: ../01-MultiSurfaceEnergyFlowMaker/05_pt111_diffuse_basis.py"
    )

# Load or create slab structure
slab_file = source_dir / "pt111_slab.cif"
if slab_file.exists():
    pt111_slab = Structure.from_file(str(slab_file))
    print(f"Loaded slab structure from {slab_file}")
else:
    # Recreate Pt(111) slab (same as used in MultiSurfaceEnergyFlowMaker)
    from pymatgen.core import Lattice

    print("Recreating Pt(111) slab structure...")
    a = 3.92  # Pt lattice constant
    lattice = Lattice.from_parameters(
        a=a * 3**0.5, b=a * 3, c=20.0, alpha=90, beta=90, gamma=90
    )
    pt_coords = []
    for layer in range(4):
        z = layer * 2.27
        pt_coords.extend(
            [
                [0.0, 0.0, z],
                [a * 3**0.5 / 2, a / 2, z],
                [a * 3**0.5 / 2, 3 * a / 2, z],
            ]
        )
    pt111_slab = Structure(
        lattice=lattice,
        species=["Pt"] * 12,
        coords=pt_coords,
        coords_are_cartesian=True,
    )
    print(f"Created Pt(111) slab with {len(pt111_slab)} atoms")

# Load pre-calculated energies
energies_file = source_dir / energies_filename
if not energies_file.exists():
    raise FileNotFoundError(f"Energies file '{energies_file}' not found.")
with open(energies_file) as f:
    energies = json.load(f)
slab_energy = energies["slab_energy"]
print(f"Loaded slab energy: {slab_energy:.6f} eV")

# Load or generate PAO.BasisSizes for diffuse basis
basissizes_file = source_dir / "pao_basissizes.txt"
if basissizes_file.exists():
    with open(basissizes_file) as f:
        pao_basissizes = json.load(f)
    print(f"Loaded PAO.BasisSizes for {len(pao_basissizes)} species")
else:
    # Generate PAO.BasisSizes using apply_diffuse_basis_to_surface
    from atomate2.siesta.sets.utils import apply_diffuse_basis_to_surface

    species_labels, pao_basissizes, surface_info = apply_diffuse_basis_to_surface(
        pt111_slab,
        surface_basis=energies.get("surface_basis", "DZP"),
        bulk_basis=energies.get("bulk_basis", "DZ"),
        surface_layers=1,
    )
    pt111_slab.add_site_property("species_label", species_labels)
    print("Generated PAO.BasisSizes for diffuse basis")
    print(
        f"  Surface atoms: {len(surface_info['surface'])}, Bulk atoms: {len(surface_info['bulk'])}"
    )

# ============================================================================
# Create makers with tier presets
# ============================================================================
slab_maker = StaticMaker()
slab_maker = apply_tier_preset(
    slab_maker,
    "electrocatalysis_dirty",
    override_params={"%block PAO.BasisSizes": pao_basissizes},
)

adsorbate_maker = RelaxMaker.fixed_cell_relaxation()
adsorbate_maker = apply_tier_preset(adsorbate_maker, "electrocatalysis_gas_phase")

# ============================================================================
# Define additional adsorbates
# ============================================================================
oh_molecule = Molecule(["O", "H"], [[0, 0, 0], [0.96, 0, 0]])
h_atom = Molecule(["H"], [[0, 0, 0]])

adsorbates = [
    ("OH", oh_molecule),
    ("H", h_atom),
]

# ============================================================================
# Create workflows with pre-calculated slab energy
# ============================================================================
output_dir = Path("13_pt_surface_reuse_slab_energy")
output_dir.mkdir(exist_ok=True)

for name, adsorbate in adsorbates:
    print(f"\n{'=' * 60}")
    print(f"Running {name} adsorption scan (reusing slab energy)")
    print("=" * 60)

    # Create maker with precalc_slab_energy - NO slab calculation will run!
    maker = AdsorptionScanFlowMaker(
        name=f"Pt111_{name}_adsorption",
        slab_static_maker=slab_maker,
        adsorbate_static_maker=adsorbate_maker,
        grid_size=(3, 3),
        height=1.5,
        use_custodian=True,
        custodian_max_errors=10,
        # KEY PARAMETER: Use pre-calculated slab energy
        precalc_slab_energy=slab_energy,
    )

    flow = maker.make(pt111_slab, adsorbate)

    # Run workflow
    responses = run_locally(
        flow,
        create_folders=True,
        root_dir=str(output_dir / name),
        ensure_success=True,
    )

    # Extract results
    for uuid, response in responses.items():
        if hasattr(response, "output") and response.output is not None:
            output = response.output
            if hasattr(output, "best_adsorption_energy"):
                print(f"\n{name} adsorption results:")
                print(f"  Slab energy (reused): {output.slab_energy:.6f} eV")
                print(f"  {name} adsorbate energy: {output.adsorbate_energy:.6f} eV")
                print(
                    f"  Best adsorption energy: {output.best_adsorption_energy:.6f} eV"
                )
                print(
                    f"  Best site: ({output.best_site_position[0]:.3f}, {output.best_site_position[1]:.3f})"
                )

                # Save results
                result = {
                    "adsorbate": name,
                    "slab_energy": output.slab_energy,
                    "adsorbate_energy": output.adsorbate_energy,
                    "best_adsorption_energy": output.best_adsorption_energy,
                    "best_site_position": output.best_site_position,
                }
                with open(output_dir / f"{name}_results.json", "w") as f:
                    json.dump(result, f, indent=2)
                break

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Pre-calculated slab energy: {slab_energy:.6f} eV")
print(f"Adsorbates scanned: {', '.join(name for name, _ in adsorbates)}")
print("Slab calculations performed: 0 (all reused from first tutorial)")
print(f"\nResults saved to {output_dir}/")

print("\n✓ Multi-adsorbate scan complete using pre-calculated slab energy")
