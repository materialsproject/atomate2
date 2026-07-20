"""Utilities for reading and writing structures with SIESTA-specific features.

This module provides functions for handling structures with ghost atoms (used in
SIESTA vacancy calculations). Ghost atoms are preserved through CIF round-trips
using occupancy markers, and can be read from SIESTA FDF/XV files.

Functions
---------
read_cif_with_ghost : Read CIF file and reconstruct ghost atom properties
read_siesta_with_ghost : Read SIESTA FDF/XV file and preserve ghost atom properties
write_cif_with_ghost : Write CIF file with ghost atoms marked by occupancy
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def read_cif_with_ghost(
    filename: str | Path, ghost_occupancy_threshold: float = 0.01
) -> Structure:
    """
    Read CIF file and reconstruct ghost atom site properties.

    Detects ghost atoms by low occupancy values (< ghost_occupancy_threshold)
    and recreates the 'ghost_tags' and 'species_label' site properties.

    Parameters
    ----------
    filename : str or Path
        Input CIF filename.
    ghost_occupancy_threshold : float, optional
        Occupancy threshold below which atoms are considered ghosts.
        Default is 0.01 (1%).

    Returns
    -------
    Structure
        Pymatgen Structure with 'ghost_tags' and 'species_label' properties
        if ghost atoms are detected.

    Examples
    --------
    >>> structure = read_cif_with_ghost("defect.cif")
    >>> if "ghost_tags" in structure.site_properties:
    ...     print(f"Found {sum(structure.site_properties['ghost_tags'])} ghost atoms")

    Notes
    -----
    - Ghost atoms are identified by occupancy < ghost_occupancy_threshold
    - Creates 'ghost_tags' site property (bool list)
    - Creates 'species_label' site property with "_ghost" suffix for ghost atoms
    - Compatible with CIF files written by write_cif_with_ghost()
    """
    from pymatgen.core import Structure as PMGStructure

    filename = Path(filename)
    structure = PMGStructure.from_file(str(filename))

    # Check for ghost atoms (low occupancy sites)
    ghost_tags = []
    species_labels = []
    has_ghosts = False

    for site in structure:
        # Get total occupancy for this site
        if hasattr(site.species, "items"):
            # Composition-like species (e.g., {'O': 0.001})
            total_occupancy = sum(site.species.values())
        else:
            # Regular Element/Species (occupancy = 1.0)
            total_occupancy = 1.0

        is_ghost = total_occupancy < ghost_occupancy_threshold
        ghost_tags.append(is_ghost)
        has_ghosts = has_ghosts or is_ghost

        # Create species label
        # Get element symbol from species (handle both Element and Composition)
        if hasattr(site.species, "symbol"):
            symbol = site.species.symbol
        else:
            # Composition-like species - get first element
            symbol = list(site.species.keys())[0].symbol

        if is_ghost:
            species_labels.append(f"{symbol}_ghost")
        else:
            species_labels.append(symbol)

    # Add site properties if ghosts were found
    if has_ghosts:
        structure.add_site_property("ghost_tags", ghost_tags)
        structure.add_site_property("species_label", species_labels)
        logger.info(f"Read CIF with {sum(ghost_tags)} ghost atoms")
    else:
        logger.debug("Read CIF with no ghost atoms")

    return structure


def read_siesta_with_ghost(filename: str | Path, use_xv: bool = False) -> Structure:
    """
    Read SIESTA FDF/XV file and preserve ghost atom site properties.

    Reads structure from SIESTA FDF or XV file using sisl and properly sets up
    'ghost_tags' and 'species_label' site properties for ghost atoms
    (identified by negative Z in ChemicalSpeciesLabel block).

    Parameters
    ----------
    filename : str or Path
        Input FDF filename (species info is always read from FDF).
    use_xv : bool, optional
        If True, read geometry from associated XV file (final relaxed geometry).
        If False, read from FDF file directly (initial geometry).
        Default is False (read from FDF).

    Returns
    -------
    Structure
        Pymatgen Structure with 'ghost_tags' and 'species_label' properties
        set for ghost atoms.

    Examples
    --------
    >>> # Read initial geometry from FDF
    >>> structure = read_siesta_with_ghost("defect_structure.fdf")

    >>> # Read relaxed geometry from XV
    >>> structure = read_siesta_with_ghost("siesta.fdf", use_xv=True)

    >>> # Use with RelaxMaker
    >>> from atomate2.siesta.jobs.core import RelaxMaker
    >>> structure = read_siesta_with_ghost("V_S_2c_qp0/defect_structure.fdf")
    >>> maker = RelaxMaker.fixed_cell_relaxation()
    >>> job = maker.make(structure)

    Notes
    -----
    - Ghost atoms are identified by negative Z in ChemicalSpeciesLabel
    - Creates 'ghost_tags' site property (bool list)
    - Creates 'species_label' site property (preserves labels like 'S_ghost')
    - Compatible with FDF files written by defect generators
    - For XV files, pass the FDF path with use_xv=True
    """
    from atomate2.siesta.powerups import siesta_to_pymatgen

    filename = Path(filename)

    # Read structure keeping ghost atoms
    structure = siesta_to_pymatgen(
        str(filename),
        use_xv=use_xv,
        remove_ghost=False,  # Keep ghost atoms!
        add_site_properties=True,
    )

    # Add ghost_tags based on species_Z (negative Z = ghost) or species_label
    if "species_Z" in structure.site_properties:
        species_Z = structure.site_properties["species_Z"]
        ghost_tags = [Z < 0 for Z in species_Z]
    elif "species_label" in structure.site_properties:
        species_labels = structure.site_properties["species_label"]
        ghost_tags = ["_ghost" in label for label in species_labels]
    else:
        # No ghost info available
        ghost_tags = [False] * len(structure)

    structure.add_site_property("ghost_tags", ghost_tags)

    n_ghosts = sum(ghost_tags)
    if n_ghosts > 0:
        logger.info(f"Read SIESTA file with {n_ghosts} ghost atoms")
    else:
        logger.debug("Read SIESTA file with no ghost atoms")

    return structure


def write_cif_with_ghost(structure: Structure, filename: str | Path) -> None:
    """
    Write CIF file with ghost atoms properly marked.

    Ghost atoms (used in SIESTA vacancy calculations) are marked with
    occupancy = 0.001 (near-zero) in the CIF file. This allows visualization
    tools to display the ghost atom positions while clearly distinguishing
    them from regular atoms.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure to write. If it has 'ghost_tags' site property,
        ghost atoms will be marked with occupancy=0.001.
    filename : str or Path
        Output CIF filename.

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> structure = Structure.from_file("defect.cif")
    >>> write_cif_with_ghost(structure, "output.cif")

    Notes
    -----
    - Ghost atoms are identified by the 'ghost_tags' site property
    - Occupancy=0.001 is used instead of 0.0 because pymatgen removes
      sites with zero occupancy
    - Normal atoms have occupancy=1.0
    - The chemical formula will reflect the ghost atom occupancy
      (e.g., "Mg4 O3.001" for a structure with one O ghost atom)
    """
    from pymatgen.core import PeriodicSite, Structure as PMGStructure

    filename = Path(filename)

    # Check if structure has ghost atoms
    if "ghost_tags" in structure.site_properties:
        # Rebuild structure with occupancy in species composition
        new_sites = []
        for i, site in enumerate(structure):
            is_ghost = structure.site_properties["ghost_tags"][i]

            # Get element symbol - handle both Element and Composition (fractional occupancy)
            if hasattr(site.species, "symbol"):
                symbol = site.species.symbol
            else:
                # Composition (fractional occupancy) - get first element
                symbol = list(site.species.keys())[0].symbol

            if is_ghost:
                # Ghost atom: use composition dict with occupancy=0.001
                # Can't use 0.0 because pymatgen removes sites with zero occupancy
                species = {symbol: 0.001}
            else:
                # Normal atom: occupancy=1.0
                species = symbol

            new_site = PeriodicSite(
                species,
                site.frac_coords,
                structure.lattice,
                properties=site.properties,
            )
            new_sites.append(new_site)

        structure_to_write = PMGStructure.from_sites(new_sites)
        logger.info(
            f"Writing CIF with {sum(structure.site_properties['ghost_tags'])} ghost atoms"
        )
    else:
        # No ghost atoms, write structure as-is
        structure_to_write = structure

    # Write CIF
    structure_to_write.to(filename=str(filename), fmt="cif")
    logger.debug(f"Wrote structure to {filename}")
