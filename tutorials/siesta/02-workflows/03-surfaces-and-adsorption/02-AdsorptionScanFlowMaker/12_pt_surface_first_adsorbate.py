#!/usr/bin/env python
"""First adsorbate scan on Pt(111) - calculates and saves slab energy.

This tutorial demonstrates adsorption scanning on a Pt(111) surface with
diffuse basis for surface atoms. The slab energy is saved to a file so it
can be reused in subsequent calculations (see 13_pt_surface_reuse_slab_energy.py).

ALTERNATIVE: If you only need the clean slab energy (without adsorption),
use ../01-MultiSurfaceEnergyFlowMaker/05_pt111_diffuse_basis.py instead.

Run this tutorial first, then use the saved slab energy in the next tutorial
to avoid redundant slab calculations.

Output files:
- pt111_energies.json: Contains slab_energy, adsorbate_energy, best_adsorption_energy
- pt111_slab.cif: The Pt(111) slab structure for reference
- pao_basissizes.txt: Basis set configuration for reuse
"""

import json
from pathlib import Path

from jobflow import run_locally
from pymatgen.core import Lattice, Molecule, Structure

from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from atomate2.siesta.sets.utils import apply_diffuse_basis_to_surface

# ============================================================================
# Create Pt(111) slab
# ============================================================================
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

# ============================================================================
# Apply diffuse basis to surface atoms for better accuracy
# ============================================================================
species_labels, pao_basissizes, surface_info = apply_diffuse_basis_to_surface(
    pt111_slab,
    surface_basis="DZP",  # Larger basis for surface atoms
    bulk_basis="DZ",  # Standard basis for bulk atoms
    surface_layers=1,
)
pt111_slab.add_site_property("species_label", species_labels)

print(f"Surface atoms ({len(surface_info['surface'])}): {surface_info['surface']}")
print(f"Bulk atoms ({len(surface_info['bulk'])}): {surface_info['bulk']}")
print(f"PAO.BasisSizes: {pao_basissizes}")

# Save the slab structure for reference
output_dir = Path("12_pt_surface_first_adsorbate")
output_dir.mkdir(exist_ok=True)
pt111_slab.to(filename=str(output_dir / "pt111_slab.cif"))
print(f"Saved slab structure to {output_dir / 'pt111_slab.cif'}")

# Also save the PAO.BasisSizes for reuse
with open(output_dir / "pao_basissizes.txt", "w") as f:
    json.dump(pao_basissizes, f, indent=2)
print(f"Saved PAO.BasisSizes to {output_dir / 'pao_basissizes.txt'}")

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
# First adsorbate: O atom
# ============================================================================
o_atom = Molecule(["O"], [[0, 0, 0]])

maker = AdsorptionScanFlowMaker(
    name="Pt111_O_adsorption",
    slab_static_maker=slab_maker,
    adsorbate_static_maker=adsorbate_maker,
    grid_size=(3, 3),
    height=1.5,
    use_custodian=True,
    custodian_max_errors=10,
)

flow = maker.make(pt111_slab, o_atom)

# ============================================================================
# Run workflow
# ============================================================================
print("\nRunning O adsorption scan on Pt(111)...")
print("This calculates the clean slab energy which can be reused later.")

responses = run_locally(
    flow,
    create_folders=True,
    root_dir=str(output_dir),
    ensure_success=True,
)

# ============================================================================
# Extract and save results
# ============================================================================
# Get the output from the flow
# The flow output is an AdsorptionScanDocument
for uuid, response in responses.items():
    if hasattr(response, "output") and response.output is not None:
        output = response.output
        if hasattr(output, "slab_energy"):
            slab_energy = output.slab_energy
            adsorbate_energy = output.adsorbate_energy
            best_ads_energy = output.best_adsorption_energy

            # Save slab energy for reuse in next tutorial
            results = {
                "slab_energy": slab_energy,
                "adsorbate_energy_O": adsorbate_energy,
                "best_adsorption_energy_O": best_ads_energy,
                "slab_formula": output.slab_formula,
                "adsorbate_formula": output.adsorbate_formula,
            }

            with open(output_dir / "pt111_energies.json", "w") as f:
                json.dump(results, f, indent=2)

            print("\n" + "=" * 60)
            print("Results Summary")
            print("=" * 60)
            print(f"Slab energy: {slab_energy:.6f} eV")
            print(f"O adsorbate energy: {adsorbate_energy:.6f} eV")
            print(f"Best O adsorption energy: {best_ads_energy:.6f} eV")
            print(f"\nSaved energies to {output_dir / 'pt111_energies.json'}")
            print("\nUse this slab energy in 13_pt_surface_reuse_slab_energy.py")
            print("to scan additional adsorbates without recalculating the slab.")
            break

print("\n✓ First adsorbate scan complete")
