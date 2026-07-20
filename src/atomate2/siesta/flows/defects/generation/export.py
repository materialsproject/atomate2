"""Export generated defects to folders with structure files and metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.siesta.sets.siesta_structure_fdf import generate_structure_fdf
from atomate2.siesta.sets.utils.core import _get_site_atomic_number
from atomate2.siesta.sets.utils.structure_io import write_cif_with_ghost

logger = logging.getLogger(__name__)


def write_defects_to_folders(
    defects: list[dict],
    output_dir: str | Path = "defects",
    write_cif: bool = True,
    write_poscar: bool = False,
    write_fdf: bool = False,
    write_json: bool = True,
    write_summary: bool = True,
) -> dict[str, Path]:
    """
    Write generated defects to organized folders with structure files and metadata.

    This is useful for:
    - Pre-generating defect structures for manual inspection
    - Using defects outside the workflow framework
    - Archiving generated defects
    - Debugging and validation

    Parameters
    ----------
    defects : list[dict]
        List of defect dictionaries from any generator
        (SiestaVacancyGenerator, SiestaSubstitutionGenerator, SiestaInterstitialGenerator)
    output_dir : str or Path
        Base output directory. Default: "defects"
    write_cif : bool
        Write CIF files for structures. Default: True
    write_poscar : bool
        Write POSCAR files for structures. Default: False
    write_fdf : bool
        Write FDF input files (shows ghost atoms correctly). Default: False
    write_json : bool
        Write metadata JSON file. Default: True
    write_summary : bool
        Write summary README.txt. Default: True

    Returns
    -------
    dict[str, Path]
        Dictionary mapping defect names to their folder paths

    Examples
    --------
    Generate and export vacancy defects:

    >>> from atomate2.siesta.flows.defects.generation import SiestaVacancyGenerator
    >>> from atomate2.siesta.flows.defects.generation.export import (
    ...     write_defects_to_folders,
    ... )
    >>> generator = SiestaVacancyGenerator(structure)
    >>> defects = list(
    ...     generator.generate_defects(species="O", charge_states=[0, +1, +2])
    ... )
    >>> folders = write_defects_to_folders(defects, output_dir="O_vacancies")
    >>> # Creates: O_vacancies/V_O_q0/, O_vacancies/V_O_q+1/, etc.

    Export substitution defects:

    >>> from atomate2.siesta.flows.defects.generation import SiestaSubstitutionGenerator
    >>> generator = SiestaSubstitutionGenerator(structure)
    >>> defects = list(generator.generate_defects(species="Mg", dopants=["Li", "Na"]))
    >>> folders = write_defects_to_folders(defects, output_dir="dopants")

    Export with POSCAR files (for VASP compatibility):

    >>> folders = write_defects_to_folders(
    ...     defects, output_dir="defects", write_poscar=True
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing {len(defects)} defects to {output_dir}")

    folder_map: dict[str, Path] = {}
    used_names: dict[str, int] = {}  # Track used folder names to avoid overwrites
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("GENERATED DEFECTS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"\nTotal defects: {len(defects)}")
    summary_lines.append(f"Output directory: {output_dir.absolute()}\n")

    for i, defect in enumerate(defects):
        # Generate defect name with uniqueness check
        base_name = _generate_defect_name(defect)
        if base_name in used_names:
            used_names[base_name] += 1
            defect_name = f"{base_name}_{used_names[base_name]}"
        else:
            used_names[base_name] = 1
            defect_name = base_name
        defect_folder = output_dir / defect_name
        defect_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"  [{i + 1}/{len(defects)}] Writing {defect_name}")

        # Write defect structure
        if write_cif:
            defect_cif = defect_folder / "defect_structure.cif"
            write_cif_with_ghost(defect["structure"], defect_cif)

        if write_poscar:
            defect_poscar = defect_folder / "POSCAR_defect"
            defect["structure"].to(filename=str(defect_poscar), fmt="poscar")

        # Write host structure
        if write_cif:
            host_cif = defect_folder / "host_structure.cif"
            write_cif_with_ghost(defect["host_structure"], host_cif)

        if write_poscar:
            host_poscar = defect_folder / "POSCAR_host"
            defect["host_structure"].to(filename=str(host_poscar), fmt="poscar")

        # Write FDF files (shows ghost atoms correctly)
        if write_fdf:
            defect_fdf = defect_folder / "defect_structure.fdf"
            host_fdf = defect_folder / "host_structure.fdf"
            _write_fdf_file(defect["structure"], defect_fdf, f"Defect: {defect_name}")
            _write_fdf_file(
                defect["host_structure"], host_fdf, f"Host structure for {defect_name}"
            )

        # Write metadata
        if write_json:
            metadata = _extract_metadata(defect)
            metadata_file = defect_folder / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        # Write individual README
        readme_file = defect_folder / "README.txt"
        with open(readme_file, "w") as f:
            f.write(_generate_defect_readme(defect, defect_name))

        folder_map[defect_name] = defect_folder

        # Add to summary
        summary_lines.append(f"{i + 1:3d}. {defect_name}")
        summary_lines.append(f"     Folder: {defect_folder.name}")
        summary_lines.append(
            f"     Wyckoff: {defect.get('wyckoff', 'N/A')}, "
            f"Multiplicity: {defect.get('multiplicity', 'N/A')}"
        )
        summary_lines.append(f"     Charge: {defect.get('charge_state', 0):+d}\n")

    # Write summary README
    if write_summary:
        summary_file = output_dir / "README.txt"
        summary_lines.append("\n" + "=" * 80)
        summary_lines.append("FILE STRUCTURE:")
        summary_lines.append("=" * 80)
        summary_lines.append("Each defect folder contains:")
        if write_cif:
            summary_lines.append("  - defect_structure.cif    (defect structure)")
            summary_lines.append("  - host_structure.cif      (pristine structure)")
        if write_poscar:
            summary_lines.append(
                "  - POSCAR_defect           (defect structure, VASP format)"
            )
            summary_lines.append(
                "  - POSCAR_host             (pristine structure, VASP format)"
            )
        if write_fdf:
            summary_lines.append(
                "  - defect_structure.fdf    (SIESTA input, shows ghost atoms)"
            )
            summary_lines.append("  - host_structure.fdf      (SIESTA input)")
        if write_json:
            summary_lines.append("  - metadata.json           (defect metadata)")
        summary_lines.append("  - README.txt              (defect information)")
        summary_lines.append("=" * 80)

        with open(summary_file, "w") as f:
            f.write("\n".join(summary_lines))

        logger.info(f"Summary written to {summary_file}")

    logger.info(f"✓ Successfully wrote {len(defects)} defects to {output_dir}")

    return folder_map


def _generate_defect_name(defect: dict) -> str:
    """Generate a folder name for a defect."""
    defect_type = defect.get("defect_type", "defect")
    charge = defect.get("charge_state", 0)

    if defect_type == "vacancy":
        species = defect.get("species", "X")
        wyckoff = defect.get("wyckoff", "")
        # Format: V_O_4a_q0 (vacancy of O at Wyckoff 4a, charge 0)
        return f"V_{species}_{wyckoff}_q{charge:+d}".replace("+", "p").replace("-", "m")

    if defect_type == "substitution":
        original = defect.get("original_species", "X")
        dopant = defect.get("dopant_species", "Y")
        wyckoff = defect.get("wyckoff", "")
        # Format: Li_on_Mg_4a_q0 (Li substituting Mg at Wyckoff 4a, charge 0)
        return f"{dopant}_on_{original}_{wyckoff}_q{charge:+d}".replace(
            "+", "p"
        ).replace("-", "m")

    if defect_type == "interstitial":
        species = defect.get("species", "X")
        wyckoff = defect.get("wyckoff", "")
        # Format: Li_i_4a_q0 (Li interstitial at Wyckoff 4a, charge 0)
        return f"{species}_i_{wyckoff}_q{charge:+d}".replace("+", "p").replace("-", "m")

    return f"defect_{charge:+d}".replace("+", "p").replace("-", "m")


def _extract_metadata(defect: dict) -> dict:
    """Extract serializable metadata from defect dict."""
    metadata = {}

    # Basic info
    metadata["defect_type"] = defect.get("defect_type", "unknown")
    metadata["charge_state"] = defect.get("charge_state", 0)
    metadata["wyckoff"] = defect.get("wyckoff", "unknown")
    metadata["multiplicity"] = defect.get("multiplicity", 1)

    # Convert frac_coords to list if it's a numpy array
    frac_coords = defect.get("frac_coords", [])
    if hasattr(frac_coords, "tolist"):
        frac_coords = frac_coords.tolist()
    metadata["frac_coords"] = frac_coords

    # Type-specific info
    if defect.get("defect_type") == "vacancy":
        metadata["species"] = defect.get("species")
        metadata["use_ghost"] = defect.get("use_ghost", False)
        # Extract ghost atom information if present
        if (
            metadata["use_ghost"]
            and "ghost_tags" in defect["structure"].site_properties
        ):
            ghost_tags = defect["structure"].site_properties["ghost_tags"]
            ghost_indices = [i for i, is_ghost in enumerate(ghost_tags) if is_ghost]
            if ghost_indices:
                ghost_idx = ghost_indices[0]
                metadata["ghost_atom"] = {
                    "site_index": ghost_idx,
                    "frac_coords": defect["structure"][ghost_idx].frac_coords.tolist(),
                    "species_label": defect["structure"].site_properties[
                        "species_label"
                    ][ghost_idx],
                }
    elif defect.get("defect_type") == "substitution":
        metadata["original_species"] = defect.get("original_species")
        metadata["dopant_species"] = defect.get("dopant_species")
    elif defect.get("defect_type") == "interstitial":
        metadata["species"] = defect.get("species")

    # Supercell info
    if defect.get("supercell_matrix") is not None:
        supercell_matrix = defect.get("supercell_matrix")
        if hasattr(supercell_matrix, "tolist"):
            supercell_matrix = supercell_matrix.tolist()
        metadata["supercell_matrix"] = supercell_matrix

    # Structure info
    metadata["num_atoms_defect"] = len(defect["structure"])
    metadata["num_atoms_host"] = len(defect["host_structure"])
    metadata["composition_defect"] = str(defect["structure"].composition)
    metadata["composition_host"] = str(defect["host_structure"].composition)

    return metadata


def _generate_defect_readme(defect: dict, defect_name: str) -> str:
    """Generate README content for a defect."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"DEFECT: {defect_name}")
    lines.append("=" * 70)
    lines.append("")

    # Basic info
    lines.append("BASIC INFORMATION:")
    lines.append("-" * 70)
    lines.append(f"Defect type:      {defect.get('defect_type', 'unknown')}")
    lines.append(f"Charge state:     {defect.get('charge_state', 0):+d}")
    lines.append(f"Wyckoff position: {defect.get('wyckoff', 'unknown')}")
    lines.append(f"Multiplicity:     {defect.get('multiplicity', 1)}")
    lines.append(f"Frac. coords:     {defect.get('frac_coords', [])}")
    lines.append("")

    # Type-specific info
    if defect.get("defect_type") == "vacancy":
        lines.append("VACANCY INFORMATION:")
        lines.append("-" * 70)
        lines.append(f"Removed species:  {defect.get('species')}")
        lines.append(f"Ghost atom used:  {defect.get('use_ghost', False)}")
        lines.append("")

    elif defect.get("defect_type") == "substitution":
        lines.append("SUBSTITUTION INFORMATION:")
        lines.append("-" * 70)
        lines.append(f"Original species: {defect.get('original_species')}")
        lines.append(f"Dopant species:   {defect.get('dopant_species')}")
        lines.append("")

    elif defect.get("defect_type") == "interstitial":
        lines.append("INTERSTITIAL INFORMATION:")
        lines.append("-" * 70)
        lines.append(f"Interstitial:     {defect.get('species')}")
        lines.append("")

    # Structure info
    lines.append("STRUCTURE INFORMATION:")
    lines.append("-" * 70)
    lines.append(f"Defect atoms:     {len(defect['structure'])}")
    lines.append(f"Host atoms:       {len(defect['host_structure'])}")
    lines.append(f"Defect formula:   {defect['structure'].composition}")
    lines.append(f"Host formula:     {defect['host_structure'].composition}")

    if defect.get("supercell_matrix") is not None:
        lines.append("")
        lines.append("SUPERCELL:")
        lines.append("-" * 70)
        matrix = defect["supercell_matrix"]
        lines.append(f"Matrix: {matrix}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def _write_fdf_file(structure, fdf_path: Path, system_label: str) -> None:
    """Write FDF file showing ghost atoms correctly."""
    # Convert pymatgen Structure to ASE Atoms
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)

    # Add species information for ghost atoms if present
    if "ghost_tags" in structure.site_properties:
        species_labels = structure.site_properties["species_label"]
        species_Z = []
        for i, site in enumerate(structure):
            is_ghost = structure.site_properties["ghost_tags"][i]
            Z = _get_site_atomic_number(site)
            # Ghost atoms have negative Z
            species_Z.append(-Z if is_ghost else Z)

        atoms.info["species_labels"] = species_labels
        atoms.info["species_Z"] = species_Z

    # Write FDF using existing function
    generate_structure_fdf(atoms=atoms, output_file=str(fdf_path))
