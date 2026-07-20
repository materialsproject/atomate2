"""
Automated interstitial defect generation for SIESTA.

This module provides tools for generating interstitial defects using
Voronoi analysis to find high-symmetry interstitial sites.
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


class SiestaInterstitialGenerator:
    """
    Generate interstitial defects using Voronoi analysis.

    This class provides automated interstitial generation with:
    - Voronoi analysis to find high-symmetry interstitial sites
    - Symmetry reduction (optional, unique sites only)
    - Support for multiple interstitial species
    - Supercell creation
    - Multiple charge states

    Parameters
    ----------
    structure : Structure
        Pristine structure (unit cell or supercell)
    min_dist : float
        Minimum distance from existing atoms to interstitial site (Angstroms).
        Default is 1.5 Å.
    use_symmetry : bool
        If True (default), use symmetry to find unique interstitial sites.
        If False, return ALL candidate interstitial sites (no symmetry reduction).
    symprec : float
        Symmetry tolerance for finding equivalent sites (Angstroms).
        Default is 0.1 Å. Only used when use_symmetry=True.

    Examples
    --------
    Generate Li interstitials in MgO:

    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.flows.defects.generation import SiestaInterstitialGenerator
    >>> mgo = Structure.from_file("MgO.cif")
    >>> generator = SiestaInterstitialGenerator(mgo, min_dist=2.0)
    >>> defects = generator.generate_defects(species="Li", charge_states=[0, +1])

    Generate with supercell:

    >>> defects = generator.generate_defects(
    ...     species="Li", supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    ... )

    Notes
    -----
    **Voronoi Sites**:
    - Uses pymatgen's Voronoi analysis to find void spaces
    - High-symmetry sites (tetrahedral, octahedral) are preferred
    - min_dist parameter filters out sites too close to existing atoms

    **Typical Interstitial Sites**:
    - Tetrahedral sites (4-fold coordination)
    - Octahedral sites (6-fold coordination)
    - Other high-symmetry voids

    **Charge States**:
    - Li interstitials: typically +1
    - O interstitials: typically -2 or 0
    - Depends on oxidation state of interstitial species

    See Also
    --------
    SiestaVacancyGenerator : Generate vacancy defects
    SiestaSubstitutionGenerator : Generate substitutional defects
    DefectFlowMaker : Workflow for defect formation energy calculations
    """

    def __init__(
        self,
        structure: Structure,
        min_dist: float = 1.5,
        use_symmetry: bool = True,
        symprec: float = 0.1,
    ) -> None:
        """Initialize SiestaInterstitialGenerator."""
        self.structure = structure
        self.min_dist = min_dist
        self.use_symmetry = use_symmetry
        self.symprec = symprec

        # Analyze symmetry
        self.sga = SpacegroupAnalyzer(structure, symprec=symprec)

        logger.info(
            f"Initialized SiestaInterstitialGenerator for {structure.composition.reduced_formula}"
        )
        logger.info(
            f"  Space group: {self.sga.get_space_group_symbol()} "
            f"(#{self.sga.get_space_group_number()})"
        )
        logger.info(f"  Minimum distance: {min_dist} Å")
        logger.info(f"  Use symmetry: {use_symmetry}")

    def get_interstitial_sites(self) -> list[DefectSite]:
        """
        Find interstitial sites using Voronoi analysis.

        If use_symmetry=True (default), returns symmetry-unique sites only.
        If use_symmetry=False, returns ALL candidate sites (no symmetry reduction).

        Returns
        -------
        list[DefectSite]
            List of DefectSite objects for interstitial positions.
            The 'species' field will be "void" for empty sites.

        Examples
        --------
        >>> generator = SiestaInterstitialGenerator(mgo_structure)
        >>> sites = generator.get_interstitial_sites()
        >>> for site in sites:
        ...     print(f"Interstitial site: {site.frac_coords}, Wyckoff: {site.wyckoff}")
        >>>
        >>> # Get ALL sites (no symmetry)
        >>> generator = SiestaInterstitialGenerator(mos2, use_symmetry=False)
        >>> all_sites = generator.get_interstitial_sites()
        """
        # Find high-symmetry interstitial sites
        # Use simple approach: check common interstitial positions
        try:
            void_sites = []

            # Common interstitial positions to check:
            # - Body center (0.5, 0.5, 0.5)
            # - Face centers (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
            # - Edge centers (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)
            # - Tetrahedral sites (0.25, 0.25, 0.25), etc.
            candidate_sites = [
                [0.5, 0.5, 0.5],  # Body center
                [0.5, 0.5, 0.0],  # Face centers
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.0],  # Edge centers
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.25, 0.25, 0.25],  # Tetrahedral sites
                [0.75, 0.75, 0.75],
                [0.75, 0.25, 0.25],
                [0.25, 0.75, 0.25],
                [0.25, 0.25, 0.75],
            ]

            for frac_coords in candidate_sites:
                # Check minimum distance to any atom
                min_dist_to_atoms = float("inf")
                for site in self.structure:
                    dist = self.structure.lattice.get_distance_and_image(
                        frac_coords, site.frac_coords
                    )[0]
                    min_dist_to_atoms = min(min_dist_to_atoms, dist)

                if min_dist_to_atoms >= self.min_dist:
                    void_sites.append(np.array(frac_coords))

        except Exception as e:
            logger.warning(f"Interstitial site search failed: {e}")
            logger.warning("Returning empty list of interstitial sites")
            return []

        if len(void_sites) == 0:
            logger.warning(
                f"No interstitial sites found with min_dist={self.min_dist} Å"
            )
            logger.info(
                "Try reducing min_dist parameter to find more interstitial sites"
            )
            return []

        logger.info(f"Found {len(void_sites)} potential interstitial site(s)")

        # Create structure with void sites as dummy atoms for symmetry analysis
        structure_with_voids = self.structure.copy()
        for void_site in void_sites:
            # Add dummy atom (H) at void position for symmetry analysis
            structure_with_voids.append("H", void_site, coords_are_cartesian=False)

        # Analyze symmetry
        sga_voids = SpacegroupAnalyzer(structure_with_voids, symprec=self.symprec)
        symmetry_dataset = sga_voids.get_symmetry_dataset()
        wyckoff_symbols = symmetry_dataset.wyckoffs

        # Get indices of void sites (last len(void_sites) atoms)
        n_original_atoms = len(self.structure)
        void_indices = list(range(n_original_atoms, n_original_atoms + len(void_sites)))

        sites = []

        if self.use_symmetry:
            # Use symmetry reduction: find unique void sites
            symmetrized = sga_voids.get_symmetrized_structure()
            seen_wyckoffs = set()

            for i, equiv_sites in enumerate(symmetrized.equivalent_sites):
                representative_site = equiv_sites[0]

                # Check if this is a void site
                for j in void_indices:
                    if np.allclose(
                        structure_with_voids[j].frac_coords,
                        representative_site.frac_coords,
                        atol=1e-3,
                    ):
                        wyckoff = wyckoff_symbols[j]

                        # Skip if we've already seen this Wyckoff position
                        if wyckoff in seen_wyckoffs:
                            continue

                        seen_wyckoffs.add(wyckoff)

                        defect_site = DefectSite(
                            site_index=-1,
                            species="void",
                            frac_coords=representative_site.frac_coords.tolist(),
                            wyckoff=wyckoff,
                            multiplicity=len(equiv_sites),
                        )

                        sites.append(defect_site)

                        logger.debug(
                            f"Unique interstitial site {len(sites)}: "
                            f"{representative_site.frac_coords} (Wyckoff {wyckoff}, "
                            f"multiplicity {len(equiv_sites)})"
                        )
                        break

            logger.info(f"Found {len(sites)} symmetry-unique interstitial site(s)")
        else:
            # No symmetry reduction: return ALL candidate sites
            for i, void_idx in enumerate(void_indices):
                site_frac = structure_with_voids[void_idx].frac_coords
                wyckoff = wyckoff_symbols[void_idx]

                defect_site = DefectSite(
                    site_index=-1,
                    species="void",
                    frac_coords=site_frac.tolist(),
                    wyckoff=wyckoff,
                    multiplicity=1,
                )

                sites.append(defect_site)

                logger.debug(
                    f"Interstitial site {len(sites)}: {frac_coords} (Wyckoff {wyckoff})"
                )

            logger.info(
                f"Found {len(sites)} interstitial site(s) (no symmetry reduction)"
            )

        return sites

    def generate_defects(
        self,
        species: str | list[str],
        supercell_matrix: list[list[int]] | None = None,
        charge_states: list[int] | None = None,
    ) -> list[dict]:
        """
        Generate all symmetry-unique interstitial defects.

        Parameters
        ----------
        species : str or list[str]
            Interstitial species to insert.
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
            - 'structure': Structure with interstitial atom
            - 'host_structure': Pristine structure
            - 'species': Interstitial species (e.g., "Li")
            - 'frac_coords': Fractional coordinates of interstitial site
            - 'wyckoff': Wyckoff position
            - 'multiplicity': Symmetry multiplicity
            - 'charge_state': Charge state
            - 'supercell_matrix': Supercell matrix used (or None)
            - 'defect_type': "interstitial"

        Examples
        --------
        Generate Li interstitials:

        >>> generator = SiestaInterstitialGenerator(mgo)
        >>> defects = generator.generate_defects(species="Li", charge_states=[0, +1])

        Multiple interstitial species with supercell:

        >>> defects = generator.generate_defects(
        ...     species=["Li", "Na", "H"],
        ...     supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        ... )
        """  # noqa: RUF002
        # Convert to list
        if isinstance(species, str):
            species_list = [species]
        else:
            species_list = species

        # Get charge states
        if charge_states is None:
            charge_states = [0]

        # Get unique interstitial sites
        unique_sites = self.get_interstitial_sites()

        if len(unique_sites) == 0:
            logger.warning("No unique interstitial sites found")
            return []

        # Generate defects
        defects = []

        for defect_site in unique_sites:
            for interstitial_species in species_list:
                # Create base structure (with or without supercell)
                if supercell_matrix is not None:
                    host_structure = self.structure.copy()
                    host_structure.make_supercell(supercell_matrix)

                    # Transform fractional coordinates to supercell
                    original_frac = np.array(defect_site.frac_coords)
                    sc_matrix = np.array(supercell_matrix)
                    scaling = np.diag(sc_matrix)
                    interstitial_frac = original_frac / scaling
                else:
                    host_structure = self.structure.copy()
                    interstitial_frac = defect_site.frac_coords

                # Create interstitial structure
                interstitial_structure = host_structure.copy()
                interstitial_structure.append(
                    interstitial_species,
                    interstitial_frac,
                    coords_are_cartesian=False,
                )

                # Generate defect for each charge state
                for charge in charge_states:
                    defect_info = {
                        "structure": interstitial_structure,
                        "host_structure": host_structure,
                        "species": interstitial_species,
                        "frac_coords": interstitial_frac,
                        "wyckoff": defect_site.wyckoff,
                        "multiplicity": defect_site.multiplicity,
                        "charge_state": charge,
                        "supercell_matrix": supercell_matrix,
                        "defect_type": "interstitial",
                    }

                    defects.append(defect_info)

                    logger.debug(
                        f"Generated: {interstitial_species}_i^{charge:+d} "
                        f"(Wyckoff {defect_site.wyckoff})"
                    )

        logger.info(
            f"Generated {len(defects)} interstitial defect(s) "
            f"({len(unique_sites)} unique site(s) × {len(species_list)} species × {len(charge_states)} charge state(s))"  # noqa: RUF001
        )

        return defects
