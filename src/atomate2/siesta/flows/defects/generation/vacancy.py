"""Vacancy defect generation for SIESTA with ghost atom support."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def create_vacancy_with_ghost(
    structure: Structure,
    site_index: int,
    use_ghost: bool = True,
) -> Structure:
    """
    Create vacancy with automatic ghost atom insertion.

    For SIESTA, vacancies should use ghost atoms (atoms with zero
    nuclear charge but with basis functions) instead of complete
    atom removal. This ensures proper basis set coverage and grid
    sampling at the vacancy site.

    Ghost atoms are critical for:
    - Proper convergence of charged vacancies
    - Accurate basis set completeness
    - Consistent grid sampling near defect sites

    Parameters
    ----------
    structure : Structure
        Host structure to create vacancy in
    site_index : int
        Index of atom to remove (0-based)
    use_ghost : bool
        If True, replace with ghost atom; if False, remove completely.
        Default is True (recommended for SIESTA).

    Returns
    -------
    Structure
        Vacancy structure with ghost atom (if use_ghost=True) or
        with site completely removed (if use_ghost=False)

    Examples
    --------
    Create oxygen vacancy with ghost atom (recommended):

    >>> from pymatgen.core import Structure
    >>> mgo = Structure.from_file("MgO.cif")
    >>> o_indices = [i for i, site in enumerate(mgo) if site.specie.symbol == "O"]
    >>> vacancy = create_vacancy_with_ghost(mgo, o_indices[0], use_ghost=True)
    >>> # Ghost atom is at the vacancy site with ghost_tags=True

    Create vacancy without ghost atom (not recommended for SIESTA):

    >>> vacancy = create_vacancy_with_ghost(mgo, o_indices[0], use_ghost=False)
    >>> # Site is completely removed

    Notes
    -----
    Ghost atoms in SIESTA:
    - Specified in FDF file with negative atomic number (Z < 0)
    - Contribute zero electrons but provide basis set and grid sampling
    - Automatically handled by atomate2siesta's species variants system
    - Essential for accurate charged vacancy calculations

    The ghost atom is created by:
    1. Keeping the same coordinates as the original atom
    2. Setting site property "ghost_tags" to [True]
    3. Setting site property "species_label" to "{element}_ghost"

    The FDF generation code will automatically:
    - Create a ghost species entry with negative Z
    - Assign the same pseudopotential as the real element
    - Generate ChemicalSpeciesLabel block correctly

    See Also
    --------
    SiestaVacancyGenerator : Automated vacancy generation with symmetry reduction
    DefectFlowMaker : Complete defect calculation workflow
    """
    vacancy_structure = structure.copy()
    original_site = structure[site_index]
    original_species = original_site.specie.symbol

    if use_ghost:
        logger.info(
            f"Creating vacancy at site {site_index} ({original_species}) "
            f"with ghost atom"
        )

        # Check if structure already has ghost_tags and species_label properties
        # (for creating multiple vacancies sequentially)
        if "ghost_tags" in vacancy_structure.site_properties:
            # Update existing properties
            ghost_tags = list(vacancy_structure.site_properties["ghost_tags"])
            species_labels = list(vacancy_structure.site_properties["species_label"])
        else:
            # Initialize new properties for ALL sites
            n_sites = len(vacancy_structure)
            ghost_tags = [False] * n_sites
            species_labels = [site.specie.symbol for site in vacancy_structure]

        # Mark the vacancy site as ghost
        ghost_tags[site_index] = True
        species_labels[site_index] = f"{original_species}_ghost"

        # Add/update site properties on the structure
        vacancy_structure.add_site_property("ghost_tags", ghost_tags)
        vacancy_structure.add_site_property("species_label", species_labels)

        logger.debug(
            f"Ghost atom created at fractional coords {original_site.frac_coords}"
        )
    else:
        logger.warning(
            f"Creating vacancy at site {site_index} ({original_species}) "
            f"by complete removal (not recommended for SIESTA)"
        )
        logger.warning(
            "Complete atom removal may cause basis set incompleteness "
            "and grid sampling issues in SIESTA. Consider use_ghost=True."
        )

        # Complete removal (not recommended for SIESTA)
        vacancy_structure.remove_sites([site_index])

        logger.debug(f"Site {site_index} completely removed")

    return vacancy_structure


def create_vacancy_with_ghost_from_site(
    structure: Structure,
    frac_coords: list[float],
    tolerance: float = 0.01,
    use_ghost: bool = True,
) -> tuple[Structure, int]:
    """
    Create vacancy by specifying fractional coordinates instead of index.

    Convenience function that finds the site nearest to the given
    fractional coordinates and creates a vacancy there.

    Parameters
    ----------
    structure : Structure
        Host structure to create vacancy in
    frac_coords : list[float]
        Fractional coordinates [x, y, z] of site to remove
    tolerance : float
        Tolerance for finding matching site (Angstroms).
        Default is 0.01 Å.
    use_ghost : bool
        If True, replace with ghost atom; if False, remove completely.
        Default is True (recommended for SIESTA).

    Returns
    -------
    vacancy_structure : Structure
        Vacancy structure with ghost atom (if use_ghost=True)
    site_index : int
        Index of the site that was made into a vacancy

    Raises
    ------
    ValueError
        If no site found within tolerance of specified coordinates

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> mgo = Structure.from_file("MgO.cif")
    >>> # Create vacancy at center of cell
    >>> vacancy, idx = create_vacancy_with_ghost_from_site(
    ...     mgo, [0.5, 0.5, 0.5], tolerance=0.01
    ... )
    >>> print(f"Created vacancy at site {idx}")
    """
    import numpy as np

    # Convert fractional to cartesian for distance calculation
    target_cart = structure.lattice.get_cartesian_coords(frac_coords)

    # Find nearest site
    min_dist = float("inf")
    nearest_index = -1

    for i, site in enumerate(structure):
        dist = np.linalg.norm(site.coords - target_cart)
        if dist < min_dist:
            min_dist = dist
            nearest_index = i

    if min_dist > tolerance:
        raise ValueError(
            f"No site found within {tolerance} Å of fractional coordinates "
            f"{frac_coords}. Nearest site is {min_dist:.3f} Å away."
        )

    logger.info(
        f"Found site {nearest_index} at distance {min_dist:.4f} Å "
        f"from target {frac_coords}"
    )

    vacancy_structure = create_vacancy_with_ghost(
        structure, nearest_index, use_ghost=use_ghost
    )

    return vacancy_structure, nearest_index
