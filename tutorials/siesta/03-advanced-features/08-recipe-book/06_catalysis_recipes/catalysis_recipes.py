#!/usr/bin/env python
"""Catalysis & Surface Recipes - Surface energy and adsorption calculations."""

from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.core.surface import SlabGenerator
from atomate2.siesta.recipes import RecipeBook

# Create bulk silicon structure (diamond cubic, Fd-3m)
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# ==============================================================================
# Example 1: Surface Energy
# ==============================================================================
print("Example 1: Surface Energy Calculation")
surface_flow = RecipeBook.surface_energy_workflow(
    silicon,
    miller_indices=[(0, 0, 1)],
    slab_layers=2,
    vacuum=15.0,
    user_params={"a2s_kpts": [1, 1, 1]},
)
# Uncomment to run:
# results = run_locally(surface_flow, create_folders=True)

# ==============================================================================
# Example 2: Adsorption Energy
# ==============================================================================
print("Example 2: Adsorption Energy")
# Generate slab
slab_gen = SlabGenerator(silicon, (0, 0, 1), 2, 15)
slab = slab_gen.get_slab()

# Create adsorbate molecule (H atom)
h_atom = Molecule(["H"], [[0, 0, 0]])

adsorption_flow = RecipeBook.adsorption_scanning_workflow(
    slab,
    adsorbate=h_atom,
    grid_density=(2, 2),
    height_above_surface=2.0,
    # dry_run=True,
    user_params={"a2s_kpts": [1, 1, 1]},
)
# Uncomment to run:
# results = run_locally(adsorption_flow, create_folders=True)

# Dry-run mode
print("\nRunning surface energy dry-run...")
dry_run = RecipeBook.surface_energy_workflow(
    silicon, miller_indices=[(0, 0, 1)], slab_layers=5, vacuum=15.0, dry_run=True
)
# results = run_locally(dry_run, create_folders=True)
print("✅ Check folders for SIESTA input files")
