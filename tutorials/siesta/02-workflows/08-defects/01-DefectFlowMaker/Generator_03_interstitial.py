"""SiestaInterstitialGenerator: All available options."""

from pathlib import Path
from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects.generation import (
    SiestaInterstitialGenerator,
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

# Collect all examples for export
all_interstitials = []

# Example 1: Basic usage
gen = SiestaInterstitialGenerator(structure, min_dist=1.5, symprec=0.1)
defects1 = list(gen.generate_defects(species="Li"))
all_interstitials.extend(defects1)
print(f"Example 1: {len(defects1)} Li interstitial sites (min_dist=1.5 Å)")

# Example 2: Tighter distance constraint
gen = SiestaInterstitialGenerator(structure, min_dist=2.0)
defects2 = list(gen.generate_defects(species="Li"))
all_interstitials.extend(defects2)
print(f"\nExample 2: {len(defects2)} sites with min_dist=2.0 Å")

# Example 3: Looser distance constraint
gen = SiestaInterstitialGenerator(structure, min_dist=1.0)
defects3 = list(gen.generate_defects(species="Li"))
all_interstitials.extend(defects3)
print(f"\nExample 3: {len(defects3)} sites with min_dist=1.0 Å")

# Example 4: Multiple species
gen = SiestaInterstitialGenerator(structure)
li_defects = list(gen.generate_defects(species="Li"))
h_defects = list(gen.generate_defects(species="H"))
all_interstitials.extend(li_defects)
all_interstitials.extend(h_defects)
print(f"\nExample 4: {len(li_defects)} Li + {len(h_defects)} H interstitials")

# Example 5: With supercell
supercell = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
defects5 = list(gen.generate_defects(species="Li", supercell_matrix=supercell))
all_interstitials.extend(defects5)
print(f"\nExample 5: {len(defects5)} Li in 2×2×2 supercell")
print(f"  Structure has {len(defects5[0]['structure'])} atoms")

# Example 6: Multiple charge states
defects6 = list(gen.generate_defects(species="Li", charge_states=[-1, 0, +1]))
all_interstitials.extend(defects6)
print(f"\nExample 6: {len(defects6)} Li interstitials with charges")
for d in defects6:
    print(f"  Li_i^{d['charge_state']:+d} (Wyckoff {d['wyckoff']})")

# Example 7: Effect of min_dist on number of sites
print("\nExample 7: Effect of min_dist parameter")
for dist in [1.0, 1.5, 2.0]:
    gen = SiestaInterstitialGenerator(structure, min_dist=dist)
    defs = list(gen.generate_defects(species="Li"))
    print(f"  min_dist={dist:.1f} Å → {len(defs)} sites")

# Export MgO examples to folders
write_defects_to_folders(
    all_interstitials, output_dir="interstitials_mgo_export", write_fdf=True
)
print(
    f"\nExported {len(all_interstitials)} MgO interstitials to interstitials_mgo_export/"
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

mos2_interstitials = []

# Example 8: Demonstrating use_symmetry parameter
# Interstitial sites can have symmetry equivalence
print("\nExample 8: Comparing use_symmetry=True vs use_symmetry=False")

# use_symmetry=True (default): Only symmetry-unique interstitial sites
gen_sym = SiestaInterstitialGenerator(
    mos2, min_dist=1.0, use_symmetry=True, symprec=0.1
)
defects8a = list(gen_sym.generate_defects(species="Li"))
print(f"  use_symmetry=True:  {len(defects8a)} Li interstitials (symmetry-unique)")

# use_symmetry=False: ALL interstitial sites (no symmetry reduction)
gen_nosym = SiestaInterstitialGenerator(
    mos2, min_dist=1.0, use_symmetry=False, symprec=0.1
)
defects8b = list(gen_nosym.generate_defects(species="Li"))
print(f"  use_symmetry=False: {len(defects8b)} Li interstitials (all sites)")
print("  → Use use_symmetry=False for: surface slabs, specific sites, or testing")

# Example 9: H interstitials in MoS2 (intercalation)
gen_mos2 = SiestaInterstitialGenerator(mos2, min_dist=1.0, symprec=0.1)
defects9 = list(gen_mos2.generate_defects(species="H"))
mos2_interstitials.extend(defects9)
print(f"\nExample 9: {len(defects9)} H interstitial sites (intercalation)")

# Example 10: Li interstitials (battery application)
defects10 = list(gen_mos2.generate_defects(species="Li"))
mos2_interstitials.extend(defects10)
print(f"\nExample 10: {len(defects10)} Li interstitials (battery intercalation)")

# Example 11: Effect of min_dist in layered structure
print("\nExample 11: Effect of min_dist in layered MoS2")
for dist in [0.8, 1.0, 1.5, 2.0]:
    gen = SiestaInterstitialGenerator(mos2, min_dist=dist)
    defs = list(gen.generate_defects(species="Li"))
    print(f"  min_dist={dist:.1f} Å → {len(defs)} sites")

# Example 12: Multiple intercalants
gen_mos2 = SiestaInterstitialGenerator(mos2, min_dist=1.2)
na_defects = list(gen_mos2.generate_defects(species="Na"))
k_defects = list(gen_mos2.generate_defects(species="K"))
mos2_interstitials.extend(na_defects)
mos2_interstitials.extend(k_defects)
print(f"\nExample 12: {len(na_defects)} Na + {len(k_defects)} K interstitials")

# Example 13: In 3x3x1 supercell (typical for 2D)
supercell_2d = [[3, 0, 0], [0, 3, 0], [0, 0, 1]]
defects13 = list(gen_mos2.generate_defects(species="Li", supercell_matrix=supercell_2d))
mos2_interstitials.extend(defects13)
print(f"\nExample 13: {len(defects13)} Li in 3×3×1 supercell")
if defects13:
    print(f"  Structure has {len(defects13[0]['structure'])} atoms")

# Example 14: With charge states
defects14 = list(gen_mos2.generate_defects(species="Li", charge_states=[-1, 0, +1]))
mos2_interstitials.extend(defects14)
print(f"\nExample 14: {len(defects14)} Li interstitials with charge states")

# Export MoS2 examples
write_defects_to_folders(
    mos2_interstitials, output_dir="interstitials_mos2_export", write_fdf=True
)
print(
    f"\nExported {len(mos2_interstitials)} MoS2 interstitials to interstitials_mos2_export/"
)
