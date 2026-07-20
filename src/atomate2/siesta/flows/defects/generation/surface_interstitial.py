"""
Surface-aware interstitial defect generation for slab structures.

This module provides specialized interstitial generators that understand surface
vs bulk atoms in slab geometries. Critical for adsorption, intercalation, and
surface doping calculations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core import Element

from atomate2.siesta.flows.defects.generation.surface import SurfaceVacancyGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class SurfaceInterstitialGenerator(SurfaceVacancyGenerator):
    """
    Generate interstitial defects specifically at surface sites in slab structures.

    This class extends SurfaceVacancyGenerator to create interstitial defects
    (atoms added to void sites) near surface layers. Essential for adsorption,
    intercalation, and surface doping studies.

    **Key Difference from SiestaInterstitialGenerator**:
    - `SiestaInterstitialGenerator`: Uses 3D bulk Voronoi sites
    - `SurfaceInterstitialGenerator`: Surface-aware, places atoms near surface layers

    Parameters
    ----------
    slab_structure : Structure
        Slab structure (must be oriented with surface normal along z-axis).
    surface_layers : int
        Number of surface layers for interstitial placement.
        Default is 1 (near topmost layer).
    surface_side : str
        Which surface to generate interstitials near:
        - "top": Above/below top surface (most common)
        - "bottom": Above/below bottom surface
        - "both": Near both top and bottom surfaces
        Default is "top".
    layer_tolerance : float
        Z-coordinate tolerance for grouping atoms into layers (Angstroms).
        Default is 0.7 Å (suitable for layered materials).
    interstitial_offset : float
        Distance to place interstitial from surface layer (Angstroms).
        Positive = above layer, negative = below layer.
        Default is 1.5 Å (typical for adsorbates).
    interstitial_site_type : str
        Type of interstitial site to generate:
        - "ontop": Directly above surface atom
        - "hollow": Above center of surface unit cell
        - "bridge": Above bond between two surface atoms (if applicable)
        Default is "ontop".
    min_dist_from_atoms : float
        Minimum distance from existing atoms (Angstroms).
        Default is 1.0 Å (prevents overlap).

    Examples
    --------
    **Example 1: H adsorption on MoS₂ surface (ontop site)**

    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.flows.defects.generation import (
    ...     SurfaceInterstitialGenerator,
    ... )
    >>> mos2_slab = Structure.from_file("MoS2_slab.cif")
    >>>
    >>> # Generate H atoms on top of surface S atoms
    >>> generator = SurfaceInterstitialGenerator(
    ...     slab_structure=mos2_slab,
    ...     surface_layers=1,
    ...     surface_side="top",
    ...     interstitial_offset=1.5,  # 1.5 Å above surface
    ...     interstitial_site_type="ontop",
    ... )
    >>>
    >>> # Add H interstitials
    >>> defects = generator.generate_defects(
    ...     species="S",  # Reference surface S atoms
    ...     interstitial_species="H",  # Add H atoms
    ... )

    **Example 2: Li intercalation between graphene layers**

    >>> # For layered materials, place Li between layers
    >>> generator = SurfaceInterstitialGenerator(
    ...     slab_structure=graphene_bilayer,
    ...     surface_layers=1,
    ...     interstitial_offset=-0.5,  # Below top layer (between layers)
    ...     interstitial_site_type="hollow",
    ... )
    >>>
    >>> defects = generator.generate_defects(
    ...     species="C",  # Reference C atoms
    ...     interstitial_species="Li",
    ...     charge_states=[0, +1],
    ... )

    **Example 3: O adsorption on metal surface (multiple sites)**

    >>> # Generate O atoms at all surface metal sites
    >>> generator = SurfaceInterstitialGenerator(
    ...     slab_structure=pt_slab,
    ...     surface_layers=1,
    ...     surface_side="top",
    ...     interstitial_offset=2.0,  # 2 Å above surface
    ...     use_in_plane_symmetry=False,  # All sites, not just unique
    ... )
    >>>
    >>> defects = generator.generate_defects(
    ...     species="Pt",  # Reference surface Pt atoms
    ...     interstitial_species="O",
    ...     charge_states=[-2, 0],
    ... )

    Notes
    -----
    **Interstitial Site Types**:
    - `ontop`: Directly above a surface atom (most common for adsorption)
    - `hollow`: Above center of surface cell (intercalation, bridge sites)
    - `bridge`: Midpoint between two surface atoms (future)

    **Interstitial Offset**:
    - Positive: Above surface (adsorption)
    - Negative: Below surface (intercalation, subsurface doping)
    - Typical values: 1.0-2.0 Å for adsorbates

    **Common Use Cases**:
    - H adsorption on MoS₂, graphene, metal surfaces
    - O adsorption on metal oxides (catalysis)
    - Li intercalation in layered materials
    - N, C doping near surface

    See Also
    --------
    SurfaceVacancyGenerator : Surface vacancy defects
    SurfaceSubstitutionGenerator : Surface substitution defects
    SiestaInterstitialGenerator : Bulk interstitial defects
    """

    def __init__(
        self,
        slab_structure: Structure,
        surface_layers: int = 1,
        surface_side: str = "top",
        layer_tolerance: float = 0.7,
        use_in_plane_symmetry: bool = True,
        symprec: float = 0.1,
        interstitial_offset: float = 1.5,
        interstitial_site_type: str = "ontop",
        min_dist_from_atoms: float = 1.0,
    ) -> None:
        """Initialize SurfaceInterstitialGenerator."""
        # Initialize parent (surface layer detection)
        super().__init__(
            slab_structure=slab_structure,
            surface_layers=surface_layers,
            surface_side=surface_side,
            layer_tolerance=layer_tolerance,
            use_ghost_atoms=False,  # Interstitials don't use ghost atoms
            use_in_plane_symmetry=use_in_plane_symmetry,
            symprec=symprec,
        )

        self.interstitial_offset = interstitial_offset
        self.interstitial_site_type = interstitial_site_type.lower()
        self.min_dist_from_atoms = min_dist_from_atoms

        # Validate inputs
        if self.interstitial_site_type not in ["ontop", "hollow", "bridge"]:
            raise ValueError(
                f"interstitial_site_type must be 'ontop', 'hollow', or 'bridge', "
                f"got '{interstitial_site_type}'"
            )

        logger.info(
            f"Initialized SurfaceInterstitialGenerator for "
            f"{slab_structure.composition.reduced_formula}"
        )
        logger.info(f"  Interstitial offset: {interstitial_offset} Å")
        logger.info(f"  Interstitial site type: {interstitial_site_type}")
        logger.info(f"  Minimum distance from atoms: {min_dist_from_atoms} Å")

    def _find_interlayer_gap(self, reference_layer_index: int) -> float | None:
        """
        Find the interlayer gap below the given layer for intercalation.

        For layered materials (e.g., MoS₂), finds the van der Waals gap
        between separate layers, not the spacing within a single layer.

        Parameters
        ----------
        reference_layer_index : int
            Index of the reference layer (surface layer)

        Returns
        -------
        float or None
            Z-coordinate of the interlayer gap center (Angstroms)
            None if no suitable gap found

        Notes
        -----
        Interlayer gaps are identified as z-spacing > 0.3 Å between adjacent
        atomic planes. This distinguishes van der Waals gaps from bonded
        spacing within a single layer.
        """
        # Get z-positions of all layers
        layer_z_positions = [layer.z_position for layer in self.layers]

        # Find gaps below the reference layer
        for i in range(reference_layer_index - 1, 0, -1):
            # Calculate gap between layer i and layer i-1
            gap_size = layer_z_positions[i] - layer_z_positions[i - 1]

            # Interlayer (van der Waals) gaps are typically > 2.5 Å
            # Intra-layer spacing (e.g., Mo-S bonds) is typically ~1.6 Å
            # Use 2.5 Å threshold to distinguish true layer separation
            if gap_size > 2.5:
                # Found an interlayer gap
                gap_center = (layer_z_positions[i] + layer_z_positions[i - 1]) / 2.0
                logger.debug(
                    f"Found interlayer gap between layers {i - 1} and {i}: "
                    f"{layer_z_positions[i - 1]:.2f} - {layer_z_positions[i]:.2f} Å, "
                    f"center at {gap_center:.2f} Å"
                )
                return gap_center

        logger.warning(
            f"No interlayer gap found below layer {reference_layer_index}. "
            "Structure may not be suitable for intercalation."
        )
        return None

    def generate_defects(
        self,
        species: str | list[str] | None = None,
        interstitial_species: str | Element | None = None,
        supercell_matrix: list[list[int]] | None = None,
        charge_states: list[int] | None = None,
    ) -> list[dict]:
        """
        Generate surface interstitial defects.

        Parameters
        ----------
        species : str or list[str], optional
            Surface species to use as reference for interstitial positions.
            For "ontop" site: places interstitials above these atoms.
            For "hollow" site: uses these atoms to find cell center.
            If None, use all surface species.
        interstitial_species : str or Element
            Species to add as interstitial (e.g., "H", "Li", "O").
        supercell_matrix : list[list[int]], optional
            Supercell transformation matrix (3×3).
            For slabs, use in-plane expansion: [[n,0,0], [0,m,0], [0,0,1]]
            If None, use input structure without supercell.
        charge_states : list[int], optional
            List of charge states for each interstitial.
            If None, only generate neutral defects (q=0).

        Returns
        -------
        list[dict]
            List of defect dictionaries, each containing:
            - structure: Structure with interstitial atom
            - host_structure: Pristine slab (same supercell)
            - species: Interstitial species added
            - reference_species: Surface species used as reference
            - site_index: Index where interstitial was added
            - frac_coords: Fractional coordinates of interstitial
            - cart_coords: Cartesian coordinates of interstitial
            - layer_index: Reference surface layer
            - layer_z_position: Z-coordinate of reference layer
            - interstitial_z: Z-coordinate of interstitial atom
            - is_top_surface: Boolean
            - is_bottom_surface: Boolean
            - charge_state: Charge state
            - supercell_matrix: Supercell matrix used
            - defect_type: "surface_interstitial"
            - site_type: Interstitial site type (ontop/hollow/bridge)

        Examples
        --------
        >>> generator = SurfaceInterstitialGenerator(mos2_slab)
        >>> defects = generator.generate_defects(
        ...     species="S",  # Reference S atoms
        ...     interstitial_species="H",  # Add H
        ...     charge_states=[0, -1],
        ... )
        """  # noqa: RUF002
        if interstitial_species is None:
            raise ValueError("interstitial_species must be specified")

        # Convert to Element object
        if isinstance(interstitial_species, str):
            interstitial_species = Element(interstitial_species)

        # Get charge states
        if charge_states is None:
            charge_states = [0]

        # Get surface sites (reference for interstitial positions)
        surface_sites = self.get_surface_sites(species=species)

        if len(surface_sites) == 0:
            logger.warning(f"No surface sites found for species {species}")
            return []

        # Generate defects
        defects = []

        # Track generated positions for hollow sites (to avoid duplicates)
        generated_hollow_positions = set()

        for site_info in surface_sites:
            # Create base structure (with or without supercell)
            if supercell_matrix is not None:
                host_structure = self.slab_structure.copy()
                host_structure.make_supercell(supercell_matrix)

                # Find corresponding site index in supercell
                # (Similar logic to SurfaceVacancyGenerator)
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

                reference_site_index = site_index_in_sc
                reference_structure = host_structure
            else:
                reference_structure = self.slab_structure.copy()
                reference_site_index = site_info["site_index"]

            # Calculate interstitial position
            reference_site = reference_structure[reference_site_index]
            ref_cart = reference_site.coords

            if self.interstitial_site_type == "ontop":
                # Place directly above/below reference atom
                interstitial_cart = ref_cart.copy()
                interstitial_cart[2] += self.interstitial_offset
            elif self.interstitial_site_type == "hollow":
                # Hollow site at cell center (x, y), z depends on offset sign
                lattice = reference_structure.lattice
                center_frac = np.array([0.5, 0.5, 0.0])  # Will set z below
                center_cart = lattice.get_cartesian_coords(center_frac)
                interstitial_cart = center_cart.copy()  # Initialize cart coords

                if self.interstitial_offset < 0:
                    # Negative offset: INTERCALATION (between layers)
                    # Find the interlayer gap below the surface
                    gap_z = self._find_interlayer_gap(site_info["layer_index"])

                    if gap_z is None:
                        # No gap found - skip this site
                        logger.warning(
                            "No interlayer gap found for intercalation. Skipping."
                        )
                        continue

                    # Place at gap center, with optional offset
                    interstitial_cart[2] = gap_z + self.interstitial_offset
                    logger.debug(
                        f"Intercalation site: gap center at {gap_z:.2f} Å, "
                        f"offset {self.interstitial_offset:.2f} Å → "
                        f"Li at {interstitial_cart[2]:.2f} Å"
                    )
                else:
                    # Positive offset: ADSORPTION (above surface)
                    # Use surface layer z-position
                    center_frac[2] = site_info["frac_coords"][2]
                    center_cart = lattice.get_cartesian_coords(center_frac)
                    interstitial_cart = center_cart.copy()
                    interstitial_cart[2] += self.interstitial_offset
            else:  # bridge (future implementation)
                logger.warning("Bridge site not yet implemented, using ontop")
                interstitial_cart = ref_cart.copy()
                interstitial_cart[2] += self.interstitial_offset

            # For hollow sites, check if we've already generated this position
            if self.interstitial_site_type == "hollow":
                # Create a hashable position key (rounded to 3 decimals for tolerance)
                pos_key = tuple(np.round(interstitial_cart, 3))
                if pos_key in generated_hollow_positions:
                    logger.debug(
                        f"Skipping duplicate hollow site at z={interstitial_cart[2]:.2f} Å"
                    )
                    continue
                generated_hollow_positions.add(pos_key)

            # Check minimum distance from existing atoms
            too_close = False
            for site in reference_structure:
                dist = np.linalg.norm(site.coords - interstitial_cart)
                if dist < self.min_dist_from_atoms:
                    too_close = True
                    logger.debug(
                        f"Interstitial too close to {site.specie} "
                        f"(dist={dist:.2f} Å < {self.min_dist_from_atoms} Å)"
                    )
                    break

            if too_close:
                continue

            # Convert to fractional coordinates
            interstitial_frac = reference_structure.lattice.get_fractional_coords(
                interstitial_cart
            )

            # Create structure with interstitial
            interstitial_structure = reference_structure.copy()
            interstitial_structure.append(
                species=interstitial_species,
                coords=interstitial_frac,
                coords_are_cartesian=False,
            )

            # Generate defect for each charge state
            for charge in charge_states:
                defect_info = {
                    "structure": interstitial_structure,
                    "host_structure": reference_structure,
                    "species": interstitial_species.symbol,
                    "reference_species": site_info["species"],
                    "site_index": len(reference_structure),  # New atom index
                    "frac_coords": interstitial_frac.tolist(),
                    "cart_coords": interstitial_cart.tolist(),
                    "layer_index": site_info["layer_index"],
                    "layer_z_position": site_info["layer_z_position"],
                    "interstitial_z": interstitial_cart[2],
                    "is_top_surface": site_info["is_top_surface"],
                    "is_bottom_surface": site_info["is_bottom_surface"],
                    "charge_state": charge,
                    "supercell_matrix": supercell_matrix,
                    "defect_type": "surface_interstitial",
                    "site_type": self.interstitial_site_type,
                }

                defects.append(defect_info)

                logger.debug(
                    f"Generated: {interstitial_species.symbol}_i^{charge:+d} "
                    f"in layer {site_info['layer_index']} "
                    f"(z={interstitial_cart[2]:.2f} Å, {self.interstitial_site_type} site)"
                )

        logger.info(
            f"Generated {len(defects)} surface interstitial defect(s) "
            f"({len(surface_sites)} unique site(s) × {len(charge_states)} charge state(s))"  # noqa: RUF001
        )

        return defects
