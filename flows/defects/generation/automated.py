"""
Automated defect generation for SIESTA with ghost atom support.

This module provides high-level classes that wrap pymatgen's defect
generation tools and add SIESTA-specific features like ghost atoms.

Key Features:
- Automatic symmetry reduction (unique defect sites only)
- Ghost atom insertion for vacancies (SIESTA-specific)
- Supercell optimization
- Multiple charge states
- Integration with DefectFlowMaker
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.siesta.flows.defects.generation.vacancy import create_vacancy_with_ghost

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class DefectSite:
    """
    Information about a defect site.

    Attributes
    ----------
    site_index : int
        Index of the defect site in the structure
    species : str
        Chemical species at the defect site (e.g., "O", "Mg")
    frac_coords : list[float]
        Fractional coordinates [x, y, z] of defect site
    wyckoff : str
        Wyckoff position label (e.g., "4a", "8c")
    multiplicity : int
        Number of symmetry-equivalent sites
    """

    site_index: int
    species: str
    frac_coords: list[float]
    wyckoff: str
    multiplicity: int


class SiestaVacancyGenerator:
    """
    Generate vacancy defects with SIESTA-specific ghost atoms.

    This class provides automated vacancy generation with:
    - Symmetry analysis to find unique vacancy sites (optional)
    - Automatic ghost atom insertion (SIESTA-specific)
    - Supercell creation
    - Multiple charge states

    Parameters
    ----------
    structure : Structure
        Pristine structure (unit cell or supercell)
    use_ghost_atoms : bool
        If True, replace removed atoms with ghost atoms.
        If False, completely remove atoms (not recommended for SIESTA).
        Default is True (recommended for SIESTA).
    use_symmetry : bool
        If True (default), use symmetry to find unique Wyckoff positions,
        generating one vacancy per symmetry-equivalent site group.
        If False, generate vacancies at ALL atomic sites (no symmetry reduction).
        Use False for: surface slabs, specific site selection, or testing.
    symprec : float
        Symmetry tolerance for finding equivalent sites (Angstroms).
        Default is 0.1 Å. Only used when use_symmetry=True.

    Examples
    --------
    Generate all symmetry-unique vacancies with ghost atoms:

    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.flows.defects.generation import SiestaVacancyGenerator
    >>> mgo = Structure.from_file("MgO.cif")
    >>> generator = SiestaVacancyGenerator(mgo, use_ghost_atoms=True)
    >>> for defect_info in generator.generate_defects():
    ...     print(f"Vacancy at {defect_info['species']} site")
    ...     print(f"  Wyckoff: {defect_info['wyckoff']}")
    ...     print(f"  Multiplicity: {defect_info['multiplicity']}")

    Generate vacancies with supercell:

    >>> generator = SiestaVacancyGenerator(mgo)
    >>> supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]  # 2×2×2
    >>> for defect_info in generator.generate_defects(supercell_matrix=supercell_matrix):
    ...     structure = defect_info['structure']
    ...     # structure is 2×2×2 supercell with ghost atom

    Generate vacancies for multiple charge states:

    >>> generator = SiestaVacancyGenerator(mgo)
    >>> charge_states = [0, +1, +2]
    >>> for defect_info in generator.generate_defects(charge_states=charge_states):
    ...     q = defect_info['charge_state']
    ...     print(f"V_{defect_info['species']} with charge {q:+d}")

    Generate vacancies at ALL sites (no symmetry reduction):

    >>> # Useful for slabs, supercells, or when you want all positions
    >>> generator = SiestaVacancyGenerator(mos2, use_symmetry=False)
    >>> for defect_info in generator.generate_defects(species="S"):
    ...     # Returns 4 S vacancies instead of 1 (all sites, not unique)
    ...     print(f"V_S at {defect_info['frac_coords']}")

    Notes
    -----
    **Symmetry Reduction**:
    - Uses SpacegroupAnalyzer to find symmetry-equivalent sites
    - Only generates one vacancy per unique Wyckoff position
    - Reduces number of calculations by ~2-10× for typical structures

    **Ghost Atoms (SIESTA-specific)**:
    - Ghost atoms are atoms with zero nuclear charge but with basis functions
    - Essential for proper convergence of charged vacancies
    - Maintains basis set completeness and grid sampling
    - Automatically handled by atomate2siesta's FDF generation

    **Supercells**:
    - If supercell_matrix is None, uses input structure as-is
    - For defect calculations, recommend ≥10 Å separation between periodic images
    - Typical supercell sizes: 2×2×2 (64-216 atoms), 3×3×3 (216-729 atoms)

    See Also
    --------
    create_vacancy_with_ghost : Low-level vacancy creation with ghost atoms
    DefectFlowMaker : Workflow for defect formation energy calculations
    """

    def __init__(
        self,
        structure: Structure,
        use_ghost_atoms: bool = True,
        use_symmetry: bool = True,
        symprec: float = 0.1,
    ):
        """Initialize SiestaVacancyGenerator."""
        self.structure = structure
        self.use_ghost_atoms = use_ghost_atoms
        self.use_symmetry = use_symmetry
        self.symprec = symprec

        # Analyze symmetry (always do this for Wyckoff labels, even if not using for reduction)
        self.sga = SpacegroupAnalyzer(structure, symprec=symprec)
        self.symmetrized_structure = self.sga.get_symmetrized_structure()

        logger.info(
            f"Initialized SiestaVacancyGenerator for {structure.composition.reduced_formula}"
        )
        logger.info(
            f"  Space group: {self.sga.get_space_group_symbol()} "
            f"(#{self.sga.get_space_group_number()})"
        )
        logger.info(f"  Use ghost atoms: {use_ghost_atoms}")
        logger.info(f"  Use symmetry: {use_symmetry}")

    def get_unique_sites(
        self,
        species: str | list[str] | None = None,
    ) -> list[DefectSite]:
        """
        Get sites for vacancy generation.

        If use_symmetry=True (default), returns symmetry-unique sites only.
        If use_symmetry=False, returns ALL sites (no symmetry reduction).

        Parameters
        ----------
        species : str or list[str], optional
            Filter by chemical species (e.g., "O" or ["O", "Mg"]).
            If None, include all species.

        Returns
        -------
        list[DefectSite]
            List of DefectSite objects. One per unique Wyckoff position
            if use_symmetry=True, or one per atom if use_symmetry=False.

        Examples
        --------
        >>> generator = SiestaVacancyGenerator(mgo_structure)
        >>> # Get all unique sites
        >>> all_sites = generator.get_unique_sites()
        >>> print(f"Found {len(all_sites)} unique site(s)")
        >>>
        >>> # Get only oxygen sites
        >>> o_sites = generator.get_unique_sites(species="O")
        >>> for site in o_sites:
        ...     print(f"O site: Wyckoff {site.wyckoff}, multiplicity {site.multiplicity}")
        >>>
        >>> # Get ALL sites (no symmetry)
        >>> generator = SiestaVacancyGenerator(mos2, use_symmetry=False)
        >>> all_s_sites = generator.get_unique_sites(species="S")
        >>> print(f"Found {len(all_s_sites)} S sites (all atoms)")
        """
        # Convert species to list
        if isinstance(species, str):
            species_list = [species]
        elif species is None:
            species_list = None
        else:
            species_list = species

        # Get Wyckoff information for all sites
        symmetry_dataset = self.sga.get_symmetry_dataset()
        wyckoff_symbols = symmetry_dataset.wyckoffs

        sites = []

        if self.use_symmetry:
            # Use symmetry reduction: one site per equivalent group
            for i, equiv_sites in enumerate(
                self.symmetrized_structure.equivalent_sites
            ):
                # Get representative site (first in equivalent set)
                representative_site = equiv_sites[0]
                site_species = representative_site.specie.symbol

                # Filter by species if requested
                if species_list is not None and site_species not in species_list:
                    continue

                # Find index of representative site in original structure
                site_index = None
                for j, orig_site in enumerate(self.structure):
                    if np.allclose(
                        representative_site.frac_coords,
                        orig_site.frac_coords,
                        atol=1e-3,
                    ):
                        site_index = j
                        break

                if site_index is None:
                    logger.warning(
                        f"Could not find index for site at {representative_site.frac_coords}"
                    )
                    continue

                wyckoff = wyckoff_symbols[site_index]

                # Create DefectSite
                defect_site = DefectSite(
                    site_index=site_index,
                    species=site_species,
                    frac_coords=representative_site.frac_coords.tolist(),
                    wyckoff=wyckoff,
                    multiplicity=len(equiv_sites),
                )

                sites.append(defect_site)

                logger.debug(
                    f"Unique site {len(sites)}: {site_species} at "
                    f"{representative_site.frac_coords} (Wyckoff {wyckoff}, "
                    f"multiplicity {len(equiv_sites)})"
                )

            logger.info(
                f"Found {len(sites)} symmetry-unique site(s) "
                f"for species {species_list}"
            )
        else:
            # No symmetry reduction: return ALL sites
            for site_index, site in enumerate(self.structure):
                site_species = site.specie.symbol

                # Filter by species if requested
                if species_list is not None and site_species not in species_list:
                    continue

                wyckoff = wyckoff_symbols[site_index]

                # Create DefectSite
                defect_site = DefectSite(
                    site_index=site_index,
                    species=site_species,
                    frac_coords=site.frac_coords.tolist(),
                    wyckoff=wyckoff,
                    multiplicity=1,  # Each site treated independently
                )

                sites.append(defect_site)

                logger.debug(
                    f"Site {len(sites)}: {site_species} at "
                    f"{site.frac_coords} (Wyckoff {wyckoff})"
                )

            logger.info(
                f"Found {len(sites)} site(s) for species {species_list} "
                f"(no symmetry reduction)"
            )

        return sites

    def generate_defects(
        self,
        species: str | list[str] | None = None,
        supercell_matrix: list[list[int]] | None = None,
        charge_states: list[int] | None = None,
    ) -> list[dict]:
        """
        Generate all symmetry-unique vacancy defects.

        Parameters
        ----------
        species : str or list[str], optional
            Filter by chemical species (e.g., "O" or ["O", "Mg"]).
            If None, generate vacancies for all species.
        supercell_matrix : list[list[int]], optional
            Supercell transformation matrix (3×3).
            If None, use input structure without supercell.
            Example: [[2,0,0], [0,2,0], [0,0,2]] for 2×2×2 supercell.
        charge_states : list[int], optional
            List of charge states to generate for each vacancy.
            If None, only generate neutral defects (q=0).
            Example: [0, +1, +2] for neutral, +1, and +2 charge states.

        Returns
        -------
        list[dict]
            List of dictionaries, each containing:
            - 'structure': Vacancy structure with ghost atom (if use_ghost_atoms=True)
            - 'host_structure': Pristine structure (same supercell as defect)
            - 'species': Species removed (e.g., "O")
            - 'site_index': Index of removed site in original structure
            - 'frac_coords': Fractional coordinates of vacancy site
            - 'wyckoff': Wyckoff position
            - 'multiplicity': Symmetry multiplicity
            - 'charge_state': Charge state (0 if charge_states is None)
            - 'supercell_matrix': Supercell matrix used (or None)
            - 'defect_type': "vacancy"
            - 'use_ghost': Boolean indicating if ghost atom was used

        Examples
        --------
        Generate all vacancies in MgO:

        >>> generator = SiestaVacancyGenerator(mgo)
        >>> defects = generator.generate_defects()
        >>> print(f"Generated {len(defects)} vacancy defect(s)")

        Generate O vacancies only, with 2×2×2 supercell:

        >>> defects = generator.generate_defects(
        ...     species="O",
        ...     supercell_matrix=[[2,0,0], [0,2,0], [0,0,2]]
        ... )

        Generate O vacancies with multiple charge states:

        >>> defects = generator.generate_defects(
        ...     species="O",
        ...     charge_states=[0, +1, +2]
        ... )
        >>> # Returns 3× defects (one for each charge state)
        """
        # Get charge states
        if charge_states is None:
            charge_states = [0]

        # Get unique sites
        unique_sites = self.get_unique_sites(species=species)

        if len(unique_sites) == 0:
            logger.warning(f"No unique sites found for species {species}")
            return []

        # Generate defects
        defects = []

        for defect_site in unique_sites:
            # Create base structure (with or without supercell)
            if supercell_matrix is not None:
                host_structure = self.structure.copy()
                host_structure.make_supercell(supercell_matrix)

                # Find corresponding site index in supercell
                # Use first atom of same species that matches the original site's position
                # Supercell transformation: new_frac = M^-1 @ original_frac (for origin cell)
                site_index_in_sc = None
                original_frac = np.array(defect_site.frac_coords)

                for i, site in enumerate(host_structure):
                    if site.specie.symbol == defect_site.species:
                        # Get fractional coordinates in supercell
                        sc_frac = site.frac_coords

                        # Map to unit cell coordinates (scaled by supercell matrix)
                        # For 2×2×2 supercell, original [0.5, 0.5, 0.5] maps to [0.25, 0.25, 0.25]
                        sc_matrix = np.array(supercell_matrix)
                        # Diagonal elements give scaling
                        scaling = np.diag(sc_matrix)
                        expected_sc_frac = original_frac / scaling

                        # Check if this is the corresponding site
                        frac_diff = np.abs(sc_frac - expected_sc_frac)
                        # Handle periodic boundary (frac coords wrap around)
                        frac_diff = np.minimum(frac_diff, 1.0 - frac_diff)

                        if np.allclose(frac_diff, 0.0, atol=1e-3):
                            site_index_in_sc = i
                            break

                if site_index_in_sc is None:
                    logger.warning(
                        f"Could not find site {defect_site.species} at "
                        f"{defect_site.frac_coords} in supercell. Using first {defect_site.species} atom."
                    )
                    # Fallback: use first atom of same species
                    for i, site in enumerate(host_structure):
                        if site.specie.symbol == defect_site.species:
                            site_index_in_sc = i
                            break

                if site_index_in_sc is None:
                    logger.error(f"No {defect_site.species} atoms found in supercell!")
                    continue

                site_index = site_index_in_sc
                actual_frac_coords = host_structure[site_index].frac_coords.tolist()
            else:
                host_structure = self.structure.copy()
                site_index = defect_site.site_index
                actual_frac_coords = defect_site.frac_coords

            # Create vacancy structure (with or without ghost atom)
            vacancy_structure = create_vacancy_with_ghost(
                structure=host_structure,
                site_index=site_index,
                use_ghost=self.use_ghost_atoms,
            )

            # Generate defect for each charge state
            for charge in charge_states:
                defect_info = {
                    "structure": vacancy_structure,
                    "host_structure": host_structure,
                    "species": defect_site.species,
                    "site_index": site_index,
                    "frac_coords": actual_frac_coords,
                    "wyckoff": defect_site.wyckoff,
                    "multiplicity": defect_site.multiplicity,
                    "charge_state": charge,
                    "supercell_matrix": supercell_matrix,
                    "defect_type": "vacancy",
                    "use_ghost": self.use_ghost_atoms,
                }

                defects.append(defect_info)

                logger.debug(
                    f"Generated: V_{defect_site.species}^{charge:+d} "
                    f"(Wyckoff {defect_site.wyckoff})"
                )

        logger.info(
            f"Generated {len(defects)} vacancy defect(s) "
            f"({len(unique_sites)} unique site(s) × {len(charge_states)} charge state(s))"
        )

        return defects
