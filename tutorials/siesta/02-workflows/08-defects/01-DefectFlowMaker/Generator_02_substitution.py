"""SiestaSubstitutionGenerator: All available options."""

from pathlib import Path
from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects.generation import (
    SiestaSubstitutionGenerator,
    write_defects_to_folders,
)

# =============================================================================
# Part 1: MgO (3D bulk material)
# =============================================================================
print("=" * 60)
print("Part 1: MgO (3D bulk material)")
print("=" * 60)

# Create MgO structure
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

gen = SiestaSubstitutionGenerator(structure, symprec=0.1)

# Collect all examples for export
all_substitutions = []

# Example 1: Single dopant
defects1 = list(gen.generate_defects(species="Mg", dopants="Li"))
all_substitutions.extend(defects1)
print(f"Example 1: {len(defects1)} Li→Mg substitutions")

# Example 2: Multiple dopants
defects2 = list(gen.generate_defects(species="Mg", dopants=["Li", "Na", "K"]))
all_substitutions.extend(defects2)
print(f"\nExample 2: {len(defects2)} dopants on Mg (Li/Na/K)")

# Example 3: Dopants on multiple species
defects3 = list(gen.generate_defects(species=["Mg", "O"], dopants="Li"))
all_substitutions.extend(defects3)
print(f"\nExample 3: {len(defects3)} Li on both Mg and O sites")

# Example 4: Antisite defects
defects4 = list(gen.generate_antisites())
all_substitutions.extend(defects4)
print(f"\nExample 4: {len(defects4)} antisite defects (Mg↔O)")
for d in defects4:
    print(f"  {d['dopant_species']}→{d['original_species']} (Wyckoff {d['wyckoff']})")

# Example 5: With supercell
supercell = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
defects5 = list(
    gen.generate_defects(species="O", dopants="F", supercell_matrix=supercell)
)
all_substitutions.extend(defects5)
print(f"\nExample 5: {len(defects5)} F→O in 2×2×2 supercell")
print(f"  Structure has {len(defects5[0]['structure'])} atoms")

# Example 6: Multiple charge states
defects6 = list(
    gen.generate_defects(species="Mg", dopants="Li", charge_states=[-1, 0, +1])
)
all_substitutions.extend(defects6)
print(f"\nExample 6: {len(defects6)} Li→Mg with charges")
for d in defects6:
    print(f"  Li_Mg^{d['charge_state']:+d}")

# Example 7: Antisites with supercell and charges
defects7 = list(
    gen.generate_antisites(supercell_matrix=supercell, charge_states=[0, +1, -1])
)
all_substitutions.extend(defects7)
print(f"\nExample 7: {len(defects7)} antisites in 2×2×2 with 3 charges")

# Export MgO examples to folders
write_defects_to_folders(
    all_substitutions, output_dir="substitutions_mgo_export", write_fdf=True
)
print(
    f"\nExported {len(all_substitutions)} MgO substitutions to substitutions_mgo_export/"
)

# =============================================================================
# Part 2: MoS2 (2D layered material)
# =============================================================================
print("\n" + "=" * 60)
print("Part 2: MoS2 (2D layered material)")
print("=" * 60)

# Load MoS2 from CIF file
cif_path = Path(__file__).parent.parent.parent.parent / "00-structures" / "Mos2.cif"
mos2 = Structure.from_file(cif_path)
print(f"Loaded MoS2: {mos2.composition}, {len(mos2)} atoms")

mos2_substitutions = []

# Example 8: Demonstrating use_symmetry parameter
# MoS2 has 2 S atoms in unit cell that are symmetry-equivalent
print("\nExample 8: Comparing use_symmetry=True vs use_symmetry=False")

# use_symmetry=True (default): Only symmetry-unique sites
gen_sym = SiestaSubstitutionGenerator(mos2, use_symmetry=True, symprec=0.1)
defects8a = list(gen_sym.generate_defects(species="S", dopants="Se"))
print(f"  use_symmetry=True:  {len(defects8a)} Se→S substitution (symmetry-unique)")

# use_symmetry=False: ALL sites (no symmetry reduction)
gen_nosym = SiestaSubstitutionGenerator(mos2, use_symmetry=False, symprec=0.1)
defects8b = list(gen_nosym.generate_defects(species="S", dopants="Se"))
print(f"  use_symmetry=False: {len(defects8b)} Se→S substitutions (all sites)")
print("  → Use use_symmetry=False for: surface slabs, specific sites, or testing")

# Continue with symmetry-reduced examples
gen_mos2 = SiestaSubstitutionGenerator(mos2, symprec=0.1)

# Example 9: W doping on Mo sites (common for WS2/MoS2 alloys)
defects9 = list(gen_mos2.generate_defects(species="Mo", dopants="W"))
mos2_substitutions.extend(defects9)
print(f"\nExample 9: {len(defects9)} W→Mo substitutions (WS2/MoS2 alloy)")

# Example 10: Re doping (n-type dopant)
defects10 = list(gen_mos2.generate_defects(species="Mo", dopants="Re"))
mos2_substitutions.extend(defects10)
print(f"\nExample 10: {len(defects10)} Re→Mo substitutions (n-type doping)")

# Example 11: Multiple transition metal dopants
defects11 = list(gen_mos2.generate_defects(species="Mo", dopants=["Nb", "V", "Ta"]))
mos2_substitutions.extend(defects11)
print(f"\nExample 11: {len(defects11)} TM dopants on Mo (Nb/V/Ta)")

# Example 12: Se doping on S sites (MoSSe Janus)
defects12 = list(gen_mos2.generate_defects(species="S", dopants="Se"))
mos2_substitutions.extend(defects12)
print(f"\nExample 12: {len(defects12)} Se→S substitutions (toward MoSSe Janus)")

# Example 13: Antisite defects in MoS2
defects13 = list(gen_mos2.generate_antisites())
mos2_substitutions.extend(defects13)
print(f"\nExample 13: {len(defects13)} antisite defects (Mo↔S)")
for d in defects13:
    print(f"  {d['dopant_species']}→{d['original_species']} (Wyckoff {d['wyckoff']})")

# Example 14: In 3x3x1 supercell (typical for 2D)
supercell_2d = [[3, 0, 0], [0, 3, 0], [0, 0, 1]]
defects14 = list(
    gen_mos2.generate_defects(species="Mo", dopants="W", supercell_matrix=supercell_2d)
)
mos2_substitutions.extend(defects14)
print(f"\nExample 14: {len(defects14)} W→Mo in 3×3×1 supercell")
print(f"  Structure has {len(defects14[0]['structure'])} atoms")

# Export MoS2 examples
write_defects_to_folders(
    mos2_substitutions, output_dir="substitutions_mos2_export", write_fdf=True
)
print(
    f"\nExported {len(mos2_substitutions)} MoS2 substitutions to substitutions_mos2_export/"
)
