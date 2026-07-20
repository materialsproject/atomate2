"""
Surface-aware substitution defect generation for slab structures.

This module provides specialized substitution generators that understand surface
vs bulk atoms in slab geometries. Critical for surface doping, alloying, and
catalysis studies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core import Element

from atomate2.siesta.flows.defects.generation.surface import (
    SurfaceVacancyGenerator,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class SurfaceSubstitutionGenerator(SurfaceVacancyGenerator):
    """
    Generate substitutional defects specifically at surface sites in slab structures.

    This class extends SurfaceVacancyGenerator to create substitutional defects
    (dopants, surface alloying) at surface layers only. Essential for surface
    catalysis, doping, and alloy surface studies.

    **Key Difference from SiestaSubstitutionGenerator**:
    - `SiestaSubstitutionGenerator`: Uses 3D bulk symmetry, all atomic sites
    - `SurfaceSubstitutionGenerator`: Surface-aware, substitutes only surface atoms

    Parameters
    ----------
    slab_structure : Structure
        Slab structure (must be oriented with surface normal along z-axis).
    surface_layers : int
        Number of surface layers for substitution.
        Default is 1 (topmost layer only).
    surface_side : str
        Which surface to generate substitutions on:
        - "top": Top surface only (most common)
        - "bottom": Bottom surface only
        - "both": Both top and bottom surfaces
        Default is "top".
    layer_tolerance : float
        Z-coordinate tolerance for grouping atoms into layers (Angstroms).
        Default is 0.7 Å (suitable for layered materials).
    use_in_plane_symmetry : bool
        If True, use in-plane (2D) symmetry to reduce substitution sites.
        Default is True (recommended).
    symprec : float
        Symmetry tolerance (Angstroms). Default is 0.1 Å.

    Examples
    --------
    **Example 1: Single Mo→W substitution on MoS₂ surface**

    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.flows.defects.generation import SurfaceSubstitutionGenerator
    >>> mos2_slab = Structure.from_file("MoS2_slab.cif")
    >>>
    >>> # Replace surface Mo with W (for catalysis studies)
    >>> generator = SurfaceSubstitutionGenerator(
    ...     slab_structure=mos2_slab,
    ...     surface_layers=1,  # Top layer only
    ...     surface_side="top",
    ... )
    >>>
    >>> defects = generator.generate_defects(
    ...     species="Mo",  # Replace Mo atoms
    ...     dopants=["W"],  # With W atoms
    ... )

    **Example 2: Multiple dopants (screening study)**

    >>> # Screen different transition metals on MoS₂
    >>> generator = SurfaceSubstitutionGenerator(
    ...     slab_structure=mos2_slab,
    ...     surface_layers=1,
    ...     surface_side="top",
    ... )
    >>>
    >>> defects = generator.generate_defects(
    ...     species="Mo",
    ...     dopants=["W", "Nb", "Ta", "Re"],  # Multiple dopants
    ...     charge_states=[0]  # Neutral substitutions
    ... )
    >>> # Returns 4 defects (one per dopant)

    **Example 3: Surface alloying (Pt-Pd surface)**

    >>> # Replace surface Pt with Pd for alloy catalyst
    >>> generator = SurfaceSubstitutionGenerator(
    ...     slab_structure=pt_slab,
    ...     surface_layers=1,
    ...     surface_side="top",
    ...     use_in_plane_symmetry=False,  # All surface sites
    ... )
    >>>
    >>> defects = generator.generate_defects(
    ...     species="Pt",
    ...     dopants=["Pd"],
    ... )

    **Example 4: Subsurface substitution (near-surface doping)**

    >>> # Replace atoms in top 2 layers (surface + subsurface)
    >>> generator = SurfaceSubstitutionGenerator(
    ...     slab_structure=tio2_slab,
    ...     surface_layers=2,  # Top two layers
    ...     surface_side="top",
    ... )
    >>>
    >>> defects = generator.generate_defects(
    ...     species="Ti",
    ...     dopants=["Nb"],  # Nb doping
    ...     charge_states=[-1, 0]
    ... )

    **Example 5: Antisite defects on surface**

    >>> # S on Mo site, Mo on S site (surface antisites)
    >>> generator = SurfaceSubstitutionGenerator(
    ...     slab_structure=mos2_slab,
    ...     surface_layers=1,
    ...     surface_side="top",
    ... )
    >>>
    >>> # Mo on S site
    >>> mo_on_s = generator.generate_defects(
    ...     species="S",  # Replace S
    ...     dopants=["Mo"],  # With Mo
    ... )
    >>>
    >>> # S on Mo site
    >>> s_on_mo = generator.generate_defects(
    ...     species="Mo",  # Replace Mo
    ...     dopants=["S"],  # With S
    ... )

    **Example 6: Charge states for doping**

    >>> # N doping on surface O sites (charge states matter!)
    >>> generator = SurfaceSubstitutionGenerator(
    ...     slab_structure=tio2_slab,
    ...     surface_layers=1,
    ...     surface_side="top",
    ... )
    >>>
    >>> defects = generator.generate_defects(
    ...     species="O",
    ...     dopants=["N"],
    ...     charge_states=[-1, 0, +1]  # N oxidation states
    ... )

    Notes
    -----
    **Nomenclature**:
    - Mo_S: Mo substituting on S site (Mo replaces S)
    - W_Mo: W substituting on Mo site (W dopant)
    - For antisites: use species from same structure

    **Charge States**:
    - Depends on valence difference between host and dopant
    - W_Mo (W⁶⁺ on Mo⁴⁺): typically 0 or +2
    - Nb_Ti (Nb⁵⁺ on Ti⁴⁺): typically -1
    - N_O (N³⁻ on O²⁻): typically +1

    **Common Use Cases**:
    - Single-atom catalysts (Pt→Pd, Mo→W, etc.)
    - Surface doping (N on O sites in TiO₂)
    - Surface alloying (bimetallic catalysts)
    - Antisite defects on surfaces

    See Also
    --------
    SurfaceVacancyGenerator : Surface vacancy defects
    SurfaceInterstitialGenerator : Surface interstitial defects
    SiestaSubstitutionGenerator : Bulk substitution defects
    """

    def __init__(
        self,
        slab_structure: Structure,
        surface_layers: int = 1,
        surface_side: str = "top",
        layer_tolerance: float = 0.7,
        use_in_plane_symmetry: bool = True,
        symprec: float = 0.1,
    ):
        """Initialize SurfaceSubstitutionGenerator."""
        # Initialize parent (surface layer detection)
        super().__init__(
            slab_structure=slab_structure,
            surface_layers=surface_layers,
            surface_side=surface_side,
            layer_tolerance=layer_tolerance,
            use_ghost_atoms=False,  # Substitutions don't use ghost atoms
            use_in_plane_symmetry=use_in_plane_symmetry,
            symprec=symprec,
        )

        logger.info(
            f"Initialized SurfaceSubstitutionGenerator for "
            f"{slab_structure.composition.reduced_formula}"
        )

    def generate_defects(
        self,
        species: str | list[str],
        dopants: str | list[str] | list[Element],
        supercell_matrix: list[list[int]] | None = None,
        charge_states: list[int] | None = None,
    ) -> list[dict]:
        """
        Generate surface substitutional defects.

        Parameters
        ----------
        species : str or list[str]
            Surface species to replace (e.g., "Mo", "S", or ["Mo", "S"]).
        dopants : str or list[str] or list[Element]
            Species to substitute with (e.g., "W", ["W", "Nb", "Ta"]).
            Can be Element objects or element symbols.
        supercell_matrix : list[list[int]], optional
            Supercell transformation matrix (3×3).
            For slabs, use in-plane expansion: [[n,0,0], [0,m,0], [0,0,1]]
            If None, use input structure without supercell.
        charge_states : list[int], optional
            List of charge states for each substitution.
            If None, only generate neutral defects (q=0).
            Note: Same charge states applied to all substitutions.

        Returns
        -------
        list[dict]
            List of defect dictionaries, each containing:
            - structure: Structure with substitution
            - host_structure: Pristine slab (same supercell)
            - species: Dopant species (what was added)
            - removed_species: Host species (what was removed)
            - site_index: Index of substituted site
            - frac_coords: Fractional coordinates
            - layer_index: Surface layer index
            - layer_z_position: Z-coordinate of layer
            - is_top_surface: Boolean
            - is_bottom_surface: Boolean
            - charge_state: Charge state
            - supercell_matrix: Supercell matrix used
            - defect_type: "surface_substitution"

        Examples
        --------
        >>> generator = SurfaceSubstitutionGenerator(mos2_slab)
        >>> # Single dopant
        >>> defects = generator.generate_defects(
        ...     species="Mo",
        ...     dopants="W",
        ... )
        >>>
        >>> # Multiple dopants (screening)
        >>> defects = generator.generate_defects(
        ...     species="Mo",
        ...     dopants=["W", "Nb", "Ta"],
        ...     charge_states=[0]
        ... )
        >>> # Returns 3 defects (one per dopant)
        """
        # Convert species to list
        if isinstance(species, str):
            species_list = [species]
        else:
            species_list = species

        # Convert dopants to list of Elements
        if isinstance(dopants, str):
            dopant_list = [Element(dopants)]
        elif isinstance(dopants, Element):
            dopant_list = [dopants]
        else:
            dopant_list = [d if isinstance(d, Element) else Element(d) for d in dopants]

        # Get charge states
        if charge_states is None:
            charge_states = [0]

        # Get surface sites
        surface_sites = self.get_surface_sites(species=species_list)

        if len(surface_sites) == 0:
            logger.warning(f"No surface sites found for species {species_list}")
            return []

        # Generate defects
        defects = []

        for site_info in surface_sites:
            for dopant in dopant_list:
                # Create base structure (with or without supercell)
                if supercell_matrix is not None:
                    host_structure = self.slab_structure.copy()
                    host_structure.make_supercell(supercell_matrix)

                    # Find corresponding site index in supercell
                    site_index_in_sc = None
                    original_frac = np.array(site_info["frac_coords"])

                    for i, site in enumerate(host_structure):
                        if site.specie.symbol == site_info["species"]:
                            sc_frac = site.frac_coords
                            sc_matrix = np.array(supercell_matrix)

                            if np.allclose(sc_matrix[2, :], [0, 0, 1]):
                                expected_sc_frac = original_frac.copy()
                                expected_sc_frac[0] /= sc_matrix[0, 0]
                                expected_sc_frac[1] /= sc_matrix[1, 1]
                            else:
                                scaling = np.diag(sc_matrix)
                                expected_sc_frac = original_frac / scaling

                            frac_diff = np.abs(sc_frac - expected_sc_frac)
                            frac_diff = np.minimum(frac_diff, 1.0 - frac_diff)

                            if np.allclose(frac_diff, 0.0, atol=1e-3):
                                site_index_in_sc = i
                                break

                    if site_index_in_sc is None:
                        logger.warning(
                            f"Could not find site {site_info['species']} in supercell"
                        )
                        continue

                    site_index = site_index_in_sc
                    actual_frac_coords = host_structure[site_index].frac_coords.tolist()
                else:
                    host_structure = self.slab_structure.copy()
                    site_index = site_info["site_index"]
                    actual_frac_coords = site_info["frac_coords"]

                # Create substitution structure
                substitution_structure = host_structure.copy()
                substitution_structure.replace(
                    site_index,  # Positional argument, not keyword
                    species=dopant,
                )

                # Generate defect for each charge state
                for charge in charge_states:
                    defect_info = {
                        "structure": substitution_structure,
                        "host_structure": host_structure,
                        "species": dopant.symbol,  # Dopant (what was added)
                        "removed_species": site_info[
                            "species"
                        ],  # Host (what was removed)
                        "site_index": site_index,
                        "frac_coords": actual_frac_coords,
                        "layer_index": site_info["layer_index"],
                        "layer_z_position": site_info["layer_z_position"],
                        "is_top_surface": site_info["is_top_surface"],
                        "is_bottom_surface": site_info["is_bottom_surface"],
                        "charge_state": charge,
                        "supercell_matrix": supercell_matrix,
                        "defect_type": "surface_substitution",
                    }

                    defects.append(defect_info)

                    logger.debug(
                        f"Generated: {dopant.symbol}_{site_info['species']}^{charge:+d} "
                        f"in layer {site_info['layer_index']} "
                        f"(z={site_info['layer_z_position']:.2f} Å)"
                    )

        logger.info(
            f"Generated {len(defects)} surface substitution defect(s) "
            f"({len(surface_sites)} unique site(s) × {len(dopant_list)} dopant(s) "
            f"× {len(charge_states)} charge state(s))"
        )

        return defects
