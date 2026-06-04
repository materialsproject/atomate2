#!/usr/bin/env python
"""Adsorption site scanning with different presets for slab and molecule.

This example shows how to apply different material-specific presets to slab
and adsorbate calculations in an adsorption workflow. Different calculation
types need different parameter optimizations:

- Slab: Surface-optimized parameters (in-plane k-points, surface convergence)
- Molecule: Gas-phase parameters (Γ-point only, low temperature)
- Adsorbate+Slab: Screening parameters (faster for grid scans)

Available adsorbate presets:
- molecule_gas_phase: Isolated molecule reference (Γ-point, low temp)
- adsorbate_screening: Fast screening (reduced k-points for grids)
- surface_metal/surface_semiconductor: Full-accuracy surface calculations
"""

from jobflow import run_locally
from pymatgen.core import Lattice, Molecule, Structure
from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

# Create MgO(100) slab
lattice = Lattice.from_parameters(a=4.2, b=4.2, c=19.6, alpha=90, beta=90, gamma=90)
species = ["Mg", "Mg", "O", "O", "Mg", "Mg", "O", "O"]
coords = [
    [0.0, 0.0, 0.32],
    [0.5, 0.5, 0.32],  # Mg layer
    [0.5, 0.0, 0.36],
    [0.0, 0.5, 0.36],  # O layer
    [0.0, 0.0, 0.43],
    [0.5, 0.5, 0.43],  # Mg layer
    [0.5, 0.0, 0.47],
    [0.0, 0.5, 0.47],  # O layer
]
slab = Structure(lattice, species, coords)

# Create CO molecule
molecule = Molecule(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]])

# Create separate makers for different calculation types
slab_maker = StaticMaker(use_custodian=True, custodian_max_errors=10)
adsorbate_maker = StaticMaker(use_custodian=True, custodian_max_errors=10)

# Apply different presets to each maker
slab_maker = apply_tier_preset(
    slab_maker, "adsorbate_screening"
)  # Fast screening for slab+adsorbate
adsorbate_maker = apply_tier_preset(
    adsorbate_maker, "molecule_gas_phase"
)  # Gas-phase molecule

# Create adsorption scan workflow with custom makers
flow = AdsorptionScanFlowMaker(
    dry_run=True,
    grid_size=(3, 3),
    height=2.0,
    slab_static_maker=slab_maker,  # Screening parameters for grid
    adsorbate_static_maker=adsorbate_maker,  # Gas-phase for isolated molecule
)

workflow = flow.make(slab, molecule)
results = run_locally(workflow, create_folders=True, root_dir="02_with_presets")

print("✓ Adsorption grid scan complete with custom presets")
print("\nSlab+Adsorbate calculations ('adsorbate_screening' preset):")
print("  - Tier: 'basic' (fast screening)")
print("  - kpts: [4, 4, 1] (reduced for speed)")
print("  - Mesh.Cutoff: 200 Ry")
print("  - Optimized for rapid site scanning")
print("\nIsolated molecule calculation ('molecule_gas_phase' preset):")
print("  - Tier: 'intermediate'")
print("  - kpts: [1, 1, 1] (Γ-point only)")
print("  - ElectronicTemperature: 25 K (low temp for molecules)")
print("  - Optimized for gas-phase molecules")
print("\nBenefits of different presets:")
print("  ✓ Faster grid scans (reduced k-points for slab+adsorbate)")
print("  ✓ Accurate molecule reference (proper gas-phase settings)")
print("  ✓ Optimal convergence for each calculation type")
