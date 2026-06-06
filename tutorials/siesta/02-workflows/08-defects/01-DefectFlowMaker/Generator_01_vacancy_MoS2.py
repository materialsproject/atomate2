"""SiestaVacancyGenerator: MoS2 (2D layered) examples with symmetry comparison."""

from pathlib import Path
from pymatgen.core import Structure
from atomate2.siesta.flows.defects.generation import (
    SiestaVacancyGenerator,
    write_defects_to_folders,
)

# Load MoS2 (2H phase, 6 atoms: 2 Mo + 4 S)
cif_path = Path(__file__).parent.parent.parent.parent / "00-structures" / "Mos2.cif"
mos2 = Structure.from_file(cif_path)
print(f"MoS2: {len(mos2)} atoms ({mos2.composition})")

# =============================================================================
# Example 1 & 2: use_symmetry=True vs use_symmetry=False
# =============================================================================
print("\nEx1: Symmetry comparison (unit cell)")

gen_sym = SiestaVacancyGenerator(mos2, use_ghost_atoms=True, use_symmetry=True)
gen_nosym = SiestaVacancyGenerator(mos2, use_ghost_atoms=True, use_symmetry=False)

vac_sym = list(gen_sym.generate_defects(species="S"))
vac_nosym = list(gen_nosym.generate_defects(species="S"))

print(f"  use_symmetry=True:  {len(vac_sym)} S vacancy")
print(f"  use_symmetry=False: {len(vac_nosym)} S vacancies at:")
for v in vac_nosym:
    print(f"    {[round(x, 4) for x in v['frac_coords']]}")

write_defects_to_folders(vac_sym, output_dir="01_unitcell_sym", write_fdf=True)
write_defects_to_folders(vac_nosym, output_dir="02_unitcell_nosym", write_fdf=True)
# =============================================================================
# Example 3: Supercell with symmetry (isolated defect calculation)
# =============================================================================
print("\nEx2: 3x3x1 supercell WITH symmetry (isolated defect)")

supercell_2d = [[3, 0, 0], [0, 3, 0], [0, 0, 1]]
vac_sc_sym = list(gen_sym.generate_defects(species="S", supercell_matrix=supercell_2d))
print(f"  {len(vac_sc_sym)} S vacancy, {len(vac_sc_sym[0]['structure'])} atoms")

write_defects_to_folders(vac_sc_sym, output_dir="03_supercell_sym", write_fdf=True)

# =============================================================================
# Example 4: Supercell WITHOUT symmetry (screen all unique positions)
# =============================================================================
print("\nEx3: 3x3x1 supercell WITHOUT symmetry (all unique positions)")

vac_sc_nosym = list(
    gen_nosym.generate_defects(species="S", supercell_matrix=supercell_2d)
)
print(
    f"  {len(vac_sc_nosym)} S vacancies, {len(vac_sc_nosym[0]['structure'])} atoms each"
)

write_defects_to_folders(vac_sc_nosym, output_dir="04_supercell_nosym", write_fdf=True)

# =============================================================================
# Example 5: ALL sites in supercell (for surfaces/broken symmetry)
# =============================================================================
print("\nEx4: Generate on supercell directly (ALL 36 S sites)")

mos2_supercell = mos2.copy()
mos2_supercell.make_supercell(supercell_2d)
gen_sc = SiestaVacancyGenerator(
    mos2_supercell, use_ghost_atoms=True, use_symmetry=False
)
vac_all = list(gen_sc.generate_defects(species="S"))
print(f"  {len(vac_all)} S vacancies (one per S atom in supercell)")

# Export first 4 as example
write_defects_to_folders(vac_all[:4], output_dir="05_all_sites_sample", write_fdf=True)

# =============================================================================
# Example 6: Mo vacancies with charge states
# =============================================================================
print("\nEx5: Mo vacancies with charge states")

vac_mo = list(
    gen_nosym.generate_defects(
        species="Mo", supercell_matrix=supercell_2d, charge_states=[0, -1, -2]
    )
)
print(
    f"  {len(vac_mo)} Mo vacancies (2 sites x 3 charges), {len(vac_mo[0]['structure'])} atoms"
)

write_defects_to_folders(vac_mo, output_dir="06_mo_charged", write_fdf=True)
