"""
Automated substitutional defect generation for SIESTA.

This module provides tools for generating substitutional defects (dopants, antisites)
with symmetry reduction and SIESTA-specific optimizations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.siesta.flows.defects.generation.automated import DefectSite

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class SiestaSubstitutionGenerator:
    """
    Generate substitutional defects (dopants, antisites) with symmetry reduction.

    This class provides automated substitution generation with:
    - Symmetry analysis to find unique substitution sites (optional)
    - Support for multiple dopant species
    - Antisite defects (element from same structure)
    - Supercell creation
    - Multiple charge states

    Parameters
    ----------
    structure : Structure
        Pristine structure (unit cell or supercell)
    use_symmetry : bool
        If True (default), use symmetry to find unique Wyckoff positions.
        If False, generate substitutions at ALL atomic sites (no symmetry reduction).
    symprec : float
        Symmetry tolerance for finding equivalent sites (Angstroms).
        Default is 0.1 Å. Only used when use_symmetry=True.

    Examples
    --------
    Generate Li dopant on Mg sites in MgO:

    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.flows.defects.generation import SiestaSubstitutionGenerator
    >>> mgo = Structure.from_file("MgO.cif")
    >>> generator = SiestaSubstitutionGenerator(mgo)
    >>> defects = generator.generate_defects(
    ...     species="Mg",  # Replace Mg atoms
    ...     dopants=["Li"],  # With Li
    ...     charge_states=[-1, 0],
    ... )

    Generate antisites (Mg on O site, O on Mg site):

    >>> generator = SiestaSubstitutionGenerator(mgo)
    >>> antisites = generator.generate_antisites()
    >>> # Returns Mg_O and O_Mg defects

    Generate multiple dopants:

    >>> defects = generator.generate_defects(
    ...     species="Mg",
    ...     dopants=["Li", "Na", "K"],  # Alkali metals on Mg site
    ... )

    Notes
    -----
    **Nomenclature**:
    - Mg_O: Mg substituting on O site (Mg dopant)
    - Li_Mg: Li substituting on Mg site (Li dopant)
    - For antisites: use species from same structure

    **Charge States**:
    - Recommended charges depend on oxidation states
    - Li_Mg: typically -1 (Li+ on Mg2+ site)
    - Mg_O: typically +4 (Mg2+ on O2- site)

    See Also
    --------
    SiestaVacancyGenerator : Generate vacancy defects
    DefectFlowMaker : Workflow for defect formation energy calculations
    """

    def __init__(
        self,
        structure: Structure,
        use_symmetry: bool = True,
        symprec: float = 0.1,
    ):
        """Initialize SiestaSubstitutionGenerator."""
        self.structure = structure
        self.use_symmetry = use_symmetry
        self.symprec = symprec

        # Analyze symmetry (always do this for Wyckoff labels)
        self.sga = SpacegroupAnalyzer(structure, symprec=symprec)
        self.symmetrized_structure = self.sga.get_symmetrized_structure()

        logger.info(
            f"Initialized SiestaSubstitutionGenerator for {structure.composition.reduced_formula}"
        )
        logger.info(
            f"  Space group: {self.sga.get_space_group_symbol()} "
            f"(#{self.sga.get_space_group_number()})"
        )
        logger.info(f"  Use symmetry: {use_symmetry}")

    def get_unique_sites(
        self,
        species: str | list[str] | None = None,
    ) -> list[DefectSite]:
        """
        Get sites for substitution.

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
        >>> generator = SiestaSubstitutionGenerator(mgo_structure)
        >>> mg_sites = generator.get_unique_sites(species="Mg")
        >>> for site in mg_sites:
        ...     print(
        ...         f"Mg site: Wyckoff {site.wyckoff}, multiplicity {site.multiplicity}"
        ...     )
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
                representative_site = equiv_sites[0]
                site_species = representative_site.specie.symbol

                if species_list is not None and site_species not in species_list:
                    continue

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
                f"Found {len(sites)} symmetry-unique site(s) for species {species_list}"
            )
        else:
            # No symmetry reduction: return ALL sites
            for site_index, site in enumerate(self.structure):
                site_species = site.specie.symbol

                if species_list is not None and site_species not in species_list:
                    continue

                wyckoff = wyckoff_symbols[site_index]

                defect_site = DefectSite(
                    site_index=site_index,
                    species=site_species,
                    frac_coords=site.frac_coords.tolist(),
                    wyckoff=wyckoff,
                    multiplicity=1,
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
        species: str | list[str],
        dopants: str | list[str],
        supercell_matrix: list[list[int]] | None = None,
        charge_states: list[int] | None = None,
    ) -> list[dict]:
        """
        Generate all symmetry-unique substitutional defects.

        Parameters
        ----------
        species : str or list[str]
            Species to replace (host species).
            Example: "Mg" or ["Mg", "O"]
        dopants : str or list[str]
            Dopant species (substituting species).
            Example: "Li" or ["Li", "Na", "K"]
        supercell_matrix : list[list[int]], optional
            Supercell transformation matrix (3×3).
            If None, use input structure without supercell.
        charge_states : list[int], optional
            List of charge states to generate for each defect.
            If None, only generate neutral defects (q=0).

        Returns
        -------
        list[dict]
            List of dictionaries, each containing:
            - 'structure': Substituted structure
            - 'host_structure': Pristine structure
            - 'original_species': Species replaced (e.g., "Mg")
            - 'dopant_species': Substituting species (e.g., "Li")
            - 'site_index': Index of substituted site
            - 'frac_coords': Fractional coordinates of defect site
            - 'wyckoff': Wyckoff position
            - 'multiplicity': Symmetry multiplicity
            - 'charge_state': Charge state
            - 'supercell_matrix': Supercell matrix used (or None)
            - 'defect_type': "substitution"

        Examples
        --------
        Li dopant on Mg site:

        >>> generator = SiestaSubstitutionGenerator(mgo)
        >>> defects = generator.generate_defects(
        ...     species="Mg", dopants="Li", charge_states=[-1, 0]
        ... )

        Multiple dopants on multiple sites:

        >>> defects = generator.generate_defects(
        ...     species=["Mg", "O"],
        ...     dopants=["Li", "Na"],
        ...     supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        ... )
        """
        # Convert to lists
        if isinstance(species, str):
            species_list = [species]
        else:
            species_list = species

        if isinstance(dopants, str):
            dopants_list = [dopants]
        else:
            dopants_list = dopants

        # Get charge states
        if charge_states is None:
            charge_states = [0]

        # Get unique sites for all species
        unique_sites = self.get_unique_sites(species=species_list)

        if len(unique_sites) == 0:
            logger.warning(f"No unique sites found for species {species_list}")
            return []

        # Generate defects
        defects = []

        for defect_site in unique_sites:
            for dopant in dopants_list:
                # Skip if dopant is same as original species
                if dopant == defect_site.species:
                    logger.debug(
                        f"Skipping {dopant} on {defect_site.species} site (same species)"
                    )
                    continue

                # Create base structure (with or without supercell)
                if supercell_matrix is not None:
                    host_structure = self.structure.copy()
                    host_structure.make_supercell(supercell_matrix)

                    # Find corresponding site index in supercell
                    site_index_in_sc = None
                    original_frac = np.array(defect_site.frac_coords)

                    for i, site in enumerate(host_structure):
                        if site.specie.symbol == defect_site.species:
                            sc_frac = site.frac_coords
                            sc_matrix = np.array(supercell_matrix)
                            scaling = np.diag(sc_matrix)
                            expected_sc_frac = original_frac / scaling

                            frac_diff = np.abs(sc_frac - expected_sc_frac)
                            frac_diff = np.minimum(frac_diff, 1.0 - frac_diff)

                            if np.allclose(frac_diff, 0.0, atol=1e-3):
                                site_index_in_sc = i
                                break

                    if site_index_in_sc is None:
                        logger.warning(
                            f"Could not find site {defect_site.species} in supercell. Using first {defect_site.species} atom."
                        )
                        for i, site in enumerate(host_structure):
                            if site.specie.symbol == defect_site.species:
                                site_index_in_sc = i
                                break

                    if site_index_in_sc is None:
                        logger.error(
                            f"No {defect_site.species} atoms found in supercell!"
                        )
                        continue

                    site_index = site_index_in_sc
                    actual_frac_coords = host_structure[site_index].frac_coords.tolist()
                else:
                    host_structure = self.structure.copy()
                    site_index = defect_site.site_index
                    actual_frac_coords = defect_site.frac_coords

                # Create substituted structure
                substituted_structure = host_structure.copy()
                substituted_structure.replace(site_index, dopant)

                # Generate defect for each charge state
                for charge in charge_states:
                    defect_info = {
                        "structure": substituted_structure,
                        "host_structure": host_structure,
                        "original_species": defect_site.species,
                        "dopant_species": dopant,
                        "site_index": site_index,
                        "frac_coords": actual_frac_coords,
                        "wyckoff": defect_site.wyckoff,
                        "multiplicity": defect_site.multiplicity,
                        "charge_state": charge,
                        "supercell_matrix": supercell_matrix,
                        "defect_type": "substitution",
                    }

                    defects.append(defect_info)

                    logger.debug(
                        f"Generated: {dopant}_{defect_site.species}^{charge:+d} "
                        f"(Wyckoff {defect_site.wyckoff})"
                    )

        logger.info(
            f"Generated {len(defects)} substitutional defect(s) "
            f"({len(unique_sites)} unique site(s) × {len(dopants_list)} dopant(s) × {len(charge_states)} charge state(s))"
        )

        return defects

    def generate_antisites(
        self,
        supercell_matrix: list[list[int]] | None = None,
        charge_states: list[int] | None = None,
    ) -> list[dict]:
        """
        Generate all antisite defects (atoms swapping sites).

        An antisite is a substitutional defect where an atom from the same
        structure occupies the wrong sublattice. For example, in MgO:
        - Mg_O: Mg atom on O site
        - O_Mg: O atom on Mg site

        Parameters
        ----------
        supercell_matrix : list[list[int]], optional
            Supercell transformation matrix (3×3).
        charge_states : list[int], optional
            List of charge states to generate for each defect.
            If None, only generate neutral defects (q=0).

        Returns
        -------
        list[dict]
            List of antisite defect dictionaries (same format as generate_defects)

        Examples
        --------
        Generate all antisites in MgO:

        >>> generator = SiestaSubstitutionGenerator(mgo)
        >>> antisites = generator.generate_antisites()
        >>> # Returns: Mg_O and O_Mg defects

        With supercell and charge states:

        >>> antisites = generator.generate_antisites(
        ...     supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        ...     charge_states=[0, +1, -1],
        ... )
        """
        # Get all unique species in structure
        species_in_structure = list(set(site.specie.symbol for site in self.structure))

        logger.info(f"Generating antisite defects for species: {species_in_structure}")

        # Generate substitutions where dopant is from same structure
        antisites = []

        for original_species in species_in_structure:
            # Get other species as dopants
            other_species = [s for s in species_in_structure if s != original_species]

            # Generate defects
            defects = self.generate_defects(
                species=original_species,
                dopants=other_species,
                supercell_matrix=supercell_matrix,
                charge_states=charge_states,
            )

            antisites.extend(defects)

        logger.info(f"Generated {len(antisites)} antisite defect(s)")

        return antisites
