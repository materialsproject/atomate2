#!/usr/bin/env python
"""Grid-based adsorption site scanning."""

from jobflow import run_locally
from pymatgen.core import Lattice, Molecule, Structure
from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker

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

flow = AdsorptionScanFlowMaker(
    grid_size=(2, 2),
    height=2.0,
    use_custodian=True,  # Enable automatic error handling
    custodian_max_errors=10,  # Allow up to 10 error corrections
    tier="dirty",  # Use basic tier for faster calculations
)
workflow = flow.make(slab, molecule)
results = run_locally(workflow, create_folders=True, root_dir="01_grid_scan")

print("✓ Adsorption grid scan complete")
