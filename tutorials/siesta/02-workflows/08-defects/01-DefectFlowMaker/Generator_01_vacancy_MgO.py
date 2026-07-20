"""SiestaVacancyGenerator: MgO (3D bulk) examples."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects.generation import (
    SiestaVacancyGenerator,
    write_defects_to_folders,
)

# Create MgO structure (rock salt, 8 atoms)
lattice = Lattice.cubic(4.212)
structure = Structure(
    lattice,
    ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"],
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ],
)
print(f"MgO: {len(structure)} atoms")

# Example 1: Basic (all species, ghost atoms, symmetry-reduced)
gen = SiestaVacancyGenerator(structure, use_ghost_atoms=True)
vac1 = list(gen.generate_defects())
print(f"\nEx1 Basic: {len(vac1)} vacancies")
write_defects_to_folders(vac1, output_dir="01_basic", write_fdf=True)

# Example 2: Filter by species
vac2 = list(gen.generate_defects(species="O"))
print(f"Ex2 O only: {len(vac2)} vacancies")

# Example 3: Multiple charge states
vac3 = list(gen.generate_defects(species="O", charge_states=[0, +1, +2]))
print(f"Ex3 Charged: {len(vac3)} vacancies (V_O^0, V_O^+1, V_O^+2)")
write_defects_to_folders(vac3, output_dir="03_charged", write_fdf=True)

# Example 4: 2x2x2 supercell
supercell = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
vac4 = list(gen.generate_defects(supercell_matrix=supercell, charge_states=[0, +1, -1]))
print(f"Ex4 Supercell: {len(vac4)} vacancies, {len(vac4[0]['structure'])} atoms each")
write_defects_to_folders(vac4, output_dir="04_supercell", write_fdf=True)

# Example 5: No ghost atoms (not recommended)
gen_noghost = SiestaVacancyGenerator(structure, use_ghost_atoms=False)
vac5 = list(gen_noghost.generate_defects(species="O"))
print(f"Ex5 No ghost: {len(vac5)} vacancies (not recommended for SIESTA)")

# Example 6: Custom symmetry tolerance
gen_tight = SiestaVacancyGenerator(structure, symprec=0.01)
vac6 = list(gen_tight.generate_defects())
print(f"Ex6 symprec=0.01: {len(vac6)} vacancies")
