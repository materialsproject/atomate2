"""
Tutorial: Reading Defect Structures with Ghost Atoms

This tutorial demonstrates how to read defect structures that contain ghost atoms
from CIF and FDF files, preserving the ghost atom information for use with
SIESTA calculations.

Date: 2026-01-26
Complexity: Beginner

Key Concepts:
-------------
1. Ghost atoms are used in SIESTA for vacancy calculations
2. Standard pymatgen Structure.from_file() loses ghost information
3. Use read_cif_with_ghost() for CIF files
4. Use read_siesta_with_ghost() for FDF/XV files
5. Ghost atoms are identified by:
   - CIF: occupancy < 0.01
   - FDF/XV: negative atomic number (Z < 0)

Why Ghost Atoms?
----------------
SIESTA uses ghost atoms (basis functions without nucleus) for:
- Proper basis set completeness at vacancy sites
- Better SCF convergence for charged defects
- Accurate grid sampling at vacancy positions

The ghost atom retains the basis functions of the removed atom while
having zero nuclear charge (negative Z in SIESTA input).
"""

from pathlib import Path

print("=" * 70)
print("Tutorial: Reading Defect Structures with Ghost Atoms")
print("=" * 70)

# =============================================================================
# Part 1: The Problem - Standard loading loses ghost information
# =============================================================================
print("\n1. The Problem: Standard Structure.from_file() loses ghost info")
print("-" * 70)

from pymatgen.core import Structure  # noqa: E402

# Path to example defect structures (in tutorials/00-structures/)
structures_dir = Path(__file__).parent.parent.parent.parent / "00-structures"
cif_file = structures_dir / "defect_structure.cif"
fdf_file = structures_dir / "defect_structure.fdf"

print("\nExample files:")
print(f"  CIF: {cif_file}")
print(f"  FDF: {fdf_file}")

# Standard loading - LOSES ghost information!
if cif_file.exists():
    structure_standard = Structure.from_file(str(cif_file))
    print("\nStandard Structure.from_file() result:")
    print(f"  Atoms: {len(structure_standard)}")
    print(f"  Formula: {structure_standard.composition}")
    print(
        f"  ghost_tags: {structure_standard.site_properties.get('ghost_tags', 'NOT PRESENT!')}"
    )
    print(
        f"  species_label: {structure_standard.site_properties.get('species_label', 'NOT PRESENT!')}"
    )
    print("\n  Problem: Ghost atom information is LOST!")
else:
    print(f"\n  Note: {cif_file} not found - skipping standard load demo")

# =============================================================================
# Part 2: Solution - Use read_cif_with_ghost()
# =============================================================================
print("\n2. Solution: Use read_cif_with_ghost() for CIF files")
print("-" * 70)

from atomate2.siesta.sets.utils.structure_io import read_cif_with_ghost  # noqa: E402

if cif_file.exists():
    structure_with_ghost = read_cif_with_ghost(str(cif_file))
    print("\nread_cif_with_ghost() result:")
    print(f"  Atoms: {len(structure_with_ghost)}")
    print(f"  Formula: {structure_with_ghost.composition}")

    # Check for ghost atoms
    ghost_tags = structure_with_ghost.site_properties.get("ghost_tags")
    species_labels = structure_with_ghost.site_properties.get("species_label")

    if ghost_tags:
        n_ghosts = sum(ghost_tags)
        print(f"  ghost_tags: {ghost_tags}")
        print(f"  species_label: {species_labels}")
        print(f"\n  Found {n_ghosts} ghost atom(s)")

        # Show ghost atom details
        for i, (is_ghost, label) in enumerate(zip(ghost_tags, species_labels)):
            if is_ghost:
                site = structure_with_ghost[i]
                print(f"    Ghost atom {i}: {label} at {site.frac_coords}")
    else:
        print("  No ghost atoms found in this structure")
else:
    print(f"\n  Note: {cif_file} not found - run Generator_01_vacancy_MoS2.py first")

# =============================================================================
# Part 3: Use read_siesta_with_ghost() for FDF files
# =============================================================================
print("\n3. Use read_siesta_with_ghost() for FDF/XV files")
print("-" * 70)

from atomate2.siesta.sets.utils.structure_io import read_siesta_with_ghost  # noqa: E402

if fdf_file.exists():
    # Read initial geometry from FDF
    structure_fdf = read_siesta_with_ghost(str(fdf_file), use_xv=False)
    print("\nread_siesta_with_ghost() result (from FDF):")
    print(f"  Atoms: {len(structure_fdf)}")
    print(f"  Formula: {structure_fdf.composition}")

    ghost_tags = structure_fdf.site_properties.get("ghost_tags")
    species_labels = structure_fdf.site_properties.get("species_label")
    species_Z = structure_fdf.site_properties.get("species_Z")

    if ghost_tags:
        n_ghosts = sum(ghost_tags)
        print(f"  ghost_tags: Found {n_ghosts} ghost atom(s)")
        print(f"  species_label: {species_labels}")
        print(f"  species_Z: {species_Z}")

        # Ghost atoms have negative Z in SIESTA
        print("\n  Ghost identification:")
        for i, (is_ghost, label, Z) in enumerate(
            zip(ghost_tags, species_labels, species_Z)
        ):
            if is_ghost:
                print(f"    Site {i}: {label}, Z={Z} (negative = ghost)")
    else:
        print("  No ghost atoms found")
else:
    print(f"\n  Note: {fdf_file} not found - run Generator_01_vacancy_MoS2.py first")

# =============================================================================
# Part 4: Reading XV files (relaxed geometry)
# =============================================================================
print("\n4. Reading XV files (relaxed geometry after SIESTA run)")
print("-" * 70)

print(
    """
For XV files (relaxed geometry output), use:

    structure = read_siesta_with_ghost("siesta.fdf", use_xv=True)

This reads:
- Geometry from the XV file (relaxed positions)
- Species labels from FDF file (including ghost information)

Example workflow:
    # After SIESTA relaxation completes:
    relaxed = read_siesta_with_ghost("job_001/siesta.fdf", use_xv=True)

    # Continue with another calculation:
    maker = RelaxMaker.variable_cell_relaxation()
    job = maker.make(relaxed)
"""
)

# =============================================================================
# Part 5: Using with RelaxMaker
# =============================================================================
print("\n5. Using loaded structures with RelaxMaker")
print("-" * 70)

print(
    """
Once you have a structure with ghost_tags, use it directly with any maker:

    from atomate2.siesta.sets.utils.structure_io import read_cif_with_ghost
    from atomate2.siesta.jobs.core import RelaxMaker
    from atomate2.siesta.sets.tiers import apply_tier_preset
    from jobflow import run_locally

    # Load structure with ghost atoms
    structure = read_cif_with_ghost("V_S_2c_qp0/defect_structure.cif")

    # Create maker with appropriate preset
    maker = RelaxMaker.fixed_cell_relaxation()
    maker = apply_tier_preset(maker, "2d_vdw")  # Good for MoS2

    # Create and run job
    job = maker.make(structure)
    results = run_locally(job, create_folders=True)

The generated FDF will correctly include:
    %block ChemicalSpeciesLabel
        1  42  Mo
        2  16  S
        3 -16  S_ghost    <- Negative Z for ghost!
    %endblock ChemicalSpeciesLabel
"""
)

# =============================================================================
# Part 6: Quick Reference
# =============================================================================
print("\n6. Quick Reference")
print("-" * 70)

print(
    """
| File Type | Function                    | Example                                      |
|-----------|-----------------------------|----------------------------------------------|
| CIF       | read_cif_with_ghost()       | read_cif_with_ghost("defect.cif")            |
| FDF       | read_siesta_with_ghost()    | read_siesta_with_ghost("defect.fdf")         |
| XV        | read_siesta_with_ghost()    | read_siesta_with_ghost("siesta.fdf", use_xv=True) |

Key site properties set by these functions:
- ghost_tags: List[bool] - True for ghost atoms
- species_label: List[str] - Labels like "S", "S_ghost", "Mo"
- species_Z: List[int] - Atomic numbers (negative for ghosts in FDF/XV)
"""
)

# =============================================================================
# Part 7: Workflow Example
# =============================================================================
print("\n7. Complete Workflow Example")
print("-" * 70)

print(
    """
# Step 1: Generate defect structures
from atomate2.siesta.flows.defects.generation import (
    SiestaVacancyGenerator,
    write_defects_to_folders,
)
from pymatgen.core import Structure

structure = Structure.from_file("../../../00-structures/Mos2.cif")
generator = SiestaVacancyGenerator(structure, use_ghost_atoms=True)
defects = list(generator.generate_defects(species="S"))
write_defects_to_folders(defects, output_dir="vacancies", write_fdf=True)

# Step 2: Read generated structure with ghost atoms
from atomate2.siesta.sets.utils.structure_io import read_cif_with_ghost
defect = read_cif_with_ghost("vacancies/V_S_2c_qp0/defect_structure.cif")

# Step 3: Run calculation
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

maker = RelaxMaker.fixed_cell_relaxation()
job = maker.make(defect)
results = run_locally(job, create_folders=True)
"""
)

print("\n" + "=" * 70)
print("Tutorial Complete!")
print("=" * 70)
print("\nKey Takeaways:")
print("  1. NEVER use Structure.from_file() for defect structures with ghosts")
print("  2. Use read_cif_with_ghost() for CIF files")
print("  3. Use read_siesta_with_ghost() for FDF/XV files")
print("  4. Ghost atoms have negative Z in SIESTA (e.g., -16 for S_ghost)")
print("  5. The ghost_tags and species_label properties enable proper FDF generation")
