"""Helper functions for per-atom basis specification.

This module provides utilities to convert per-atom basis dictionaries
into species labels and PAO.BasisSizes blocks.

Phase 3: Per-Atom Basis Dictionary (Expert Mode)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure

__all__ = [
    "apply_diffuse_basis_to_surface",
    "apply_per_atom_basis",
    "create_per_atom_basis_dict",
    "detect_surface_atoms",
]


def apply_per_atom_basis(
    structure: Structure,
    per_atom_basis: dict[int, str],
    fallback_basis: str = "DZP",
) -> tuple[list[str], dict[str, str]]:
    """
    Apply per-atom basis specification to a structure.

    This function converts a per-atom basis dictionary (1-indexed) into:
    1. Species labels for each atom (to add as site property)
    2. PAO.BasisSizes dictionary mapping species labels to basis sizes

    Parameters
    ----------
    structure : Structure
        Pymatgen structure
    per_atom_basis : dict[int, str]
        Dictionary mapping atom indices (1-indexed) to basis sizes.
        Example: {1: 'TZP', 2: 'TZP', 3: 'DZP'}
    fallback_basis : str
        Basis size for atoms not in per_atom_basis (default: 'DZP')

    Returns
    -------
    species_labels : list[str]
        Species label for each atom (same length as structure)
    pao_basissizes : dict[str, str]
        Dictionary for %block PAO.BasisSizes

    Examples
    --------
    >>> from pymatgen.core import Structure, Lattice
    >>> import numpy as np
    >>>
    >>> # Create simple structure: Si + 3 O atoms
    >>> lattice = Lattice.cubic(5.0)
    >>> structure = Structure(
    ...     lattice,
    ...     ["Si", "O", "O", "O"],
    ...     [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
    ... )
    >>>
    >>> # Specify per-atom basis (1-indexed)
    >>> per_atom_basis = {
    ...     1: "TZP",  # Si: highest accuracy
    ...     2: "TZP",  # O atom 1: high accuracy
    ...     3: "DZP",  # O atom 2: medium accuracy
    ...     # Atom 4 uses fallback (DZ)
    ... }
    >>>
    >>> # Apply to structure
    >>> species_labels, pao_basissizes = apply_per_atom_basis(
    ...     structure, per_atom_basis, fallback_basis="DZ"
    ... )
    >>>
    >>> # Add to structure
    >>> structure.add_site_property("species_label", species_labels)
    >>>
    >>> # Use in RelaxMaker
    >>> from atomate2.siesta.jobs.core import RelaxMaker
    >>> maker = RelaxMaker.fixed_cell_relaxation(
    ...     user_params={
    ...         "%block PAO.BasisSizes": pao_basissizes,
    ...         "Mesh.Cutoff": "300 Ry",
    ...     }
    ... )
    >>> job = maker.make(structure)

    Notes
    -----
    - Atom indices are 1-indexed (like SIESTA)
    - Creates unique species labels for each (element, basis) combination
    - Atoms with same element and basis share species label (efficiency)
    - Validates indices are within structure size
    """
    # Validate indices
    n_atoms = len(structure)
    for atom_idx in per_atom_basis:
        if atom_idx < 1 or atom_idx > n_atoms:
            raise ValueError(
                f"Invalid atom index {atom_idx}. "
                f"Structure has {n_atoms} atoms (indices 1 to {n_atoms})"
            )

    # Create species labels for each atom
    species_labels = []
    species_basis_map = {}  # Maps (element, basis) -> species_label
    species_counter = {}  # Counter for each element
    pao_basissizes = {}

    for atom_idx, site in enumerate(structure, start=1):
        element = site.species_string

        # Get basis for this atom (from dict or fallback)
        basis = per_atom_basis.get(atom_idx, fallback_basis)

        # Create unique species label for this (element, basis) combo
        key = (element, basis)
        if key not in species_basis_map:
            # First time seeing this combination
            if element not in species_counter:
                species_counter[element] = 0

            species_counter[element] += 1

            # Create label
            if species_counter[element] == 1:
                # First variant: just use element name
                label = element
            else:
                # Additional variants: element_basis
                # Use basis as suffix to make it clear
                basis_short = basis.replace("P", "p").lower()  # dzp -> dzp
                label = f"{element}_{basis_short}"

            species_basis_map[key] = label
            pao_basissizes[label] = basis

        # Assign label to this atom
        species_labels.append(species_basis_map[key])

    return species_labels, pao_basissizes


def create_per_atom_basis_dict(
    structure: Structure,
    atom_groups: dict[str, tuple[list[int], str]],
    fallback_basis: str = "DZP",
) -> tuple[list[str], dict[str, str]]:
    """
    Create per-atom basis from grouped atom specifications.

    This is a convenience function for creating per-atom basis when you
    have logical groups of atoms (e.g., "surface", "bulk", "interface").

    Parameters
    ----------
    structure : Structure
        Pymatgen structure
    atom_groups : dict[str, tuple[list[int], str]]
        Dictionary mapping group names to (atom_indices, basis_size).
        Example: {
            "surface": ([1, 2], "TZP"),
            "bulk": ([3, 4, 5], "DZ"),
        }
    fallback_basis : str
        Basis for atoms not in any group

    Returns
    -------
    species_labels : list[str]
        Species label for each atom
    pao_basissizes : dict[str, str]
        Dictionary for %block PAO.BasisSizes

    Examples
    --------
    >>> # Define atom groups by layer
    >>> atom_groups = {
    ...     "surface": ([1, 2, 3, 4], "TZP"),  # Atoms 1-4: surface
    ...     "subsurface": ([5, 6, 7, 8], "DZP"),  # Atoms 5-8: subsurface
    ...     "bulk": ([9, 10, 11, 12], "DZ"),  # Atoms 9-12: bulk
    ... }
    >>>
    >>> species_labels, pao_basissizes = create_per_atom_basis_dict(
    ...     structure, atom_groups
    ... )
    """
    # Convert groups to per-atom dict
    per_atom_basis: dict[int, str] = {}
    for group_name, (atom_indices, basis) in atom_groups.items():
        for atom_idx in atom_indices:
            if atom_idx in per_atom_basis:
                raise ValueError(
                    f"Atom {atom_idx} appears in multiple groups: "
                    f"already in group with basis {per_atom_basis[atom_idx]}, "
                    f"now also in '{group_name}' with basis {basis}"
                )
            per_atom_basis[atom_idx] = basis

    # Use main function
    return apply_per_atom_basis(structure, per_atom_basis, fallback_basis)


def detect_surface_atoms(
    structure: Structure,
    surface_layers: int = 1,
    layer_tolerance: float = 0.5,
    vacuum_threshold: float = 5.0,
    vacuum_direction: str | None = None,
    include_both_surfaces: bool = True,
) -> dict:
    """
    Automatically detect surface atoms in a slab structure.

    This function identifies atoms at the vacuum interface by detecting
    atomic layers along the vacuum direction. Surface atoms are those
    in the outermost layer(s) adjacent to the vacuum gap.

    Parameters
    ----------
    structure : Structure
        Pymatgen structure (should be a slab with vacuum)
    surface_layers : int
        Number of atomic layers from the vacuum to consider as "surface".
        Default: 1 (only the outermost layer)
    layer_tolerance : float
        Tolerance for grouping atoms into layers (Angstroms).
        Atoms within this distance are considered in the same layer.
        Default: 0.5 Å
    vacuum_threshold : float
        Minimum gap to identify as vacuum region (Angstroms).
        Default: 5.0 Å
    vacuum_direction : str, optional
        Direction of vacuum: 'a', 'b', or 'c'. If None, automatically detected
        as the direction with the largest gap between atoms.
    include_both_surfaces : bool
        If True, include both top and bottom surfaces. If False, only top surface.
        Default: True

    Returns
    -------
    dict
        Dictionary with keys:
        - "surface": List of 1-indexed surface atom indices
        - "bulk": List of 1-indexed bulk atom indices
        - "top_surface": List of 1-indexed top surface atom indices
        - "bottom_surface": List of 1-indexed bottom surface atom indices
        - "vacuum_direction": The detected vacuum direction ('a', 'b', or 'c')
        - "n_layers": Total number of detected layers
        - "layer_positions": Z-positions of each layer (Angstroms)

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>>
    >>> # Load a slab structure
    >>> slab = Structure.from_file("Pt111_slab.cif")
    >>>
    >>> # Detect surface atoms (default: 1 outermost layer)
    >>> result = detect_surface_atoms(slab)
    >>> print(f"Surface atoms: {result['surface']}")
    >>> print(f"Bulk atoms: {result['bulk']}")
    >>>
    >>> # Include 2 outermost layers as surface
    >>> result = detect_surface_atoms(slab, surface_layers=2)
    >>>
    >>> # Specify vacuum direction explicitly
    >>> result = detect_surface_atoms(slab, vacuum_direction="c")

    Notes
    -----
    - Atom indices are 1-indexed (like SIESTA)
    - Surface detection is layer-based, not distance-based
    - Automatically finds the vacuum gap and identifies adjacent layers
    - For asymmetric slabs, you may want include_both_surfaces=False
    """
    import numpy as np

    # Get fractional coordinates
    frac_coords = structure.frac_coords

    # Determine vacuum direction if not specified
    if vacuum_direction is None:
        # Find direction with largest gap between atoms
        max_gap = 0.0
        detected_direction = "c"  # Default to c (most common for slabs)

        for i, direction in enumerate(["a", "b", "c"]):
            coords_1d = np.sort(frac_coords[:, i])
            # Calculate gaps (including wrap-around gap)
            gaps = np.diff(coords_1d)
            wrap_gap = 1.0 - coords_1d[-1] + coords_1d[0]
            all_gaps = np.append(gaps, wrap_gap)

            # Convert to Cartesian for comparison
            lattice_param = structure.lattice.abc[i]
            max_gap_this_dir = np.max(all_gaps) * lattice_param

            if max_gap_this_dir > max_gap:
                max_gap = max_gap_this_dir
                detected_direction = direction

        vacuum_direction = detected_direction

    # Get the index for the vacuum direction
    dir_idx = {"a": 0, "b": 1, "c": 2}[vacuum_direction]
    lattice_param = structure.lattice.abc[dir_idx]

    # Get Cartesian coordinates along vacuum direction
    cart_coords_vac = frac_coords[:, dir_idx] * lattice_param

    # Group atoms into layers based on their z-coordinate
    sorted_indices = np.argsort(cart_coords_vac)
    sorted_coords = cart_coords_vac[sorted_indices]

    # Identify layers by grouping atoms with similar z-coordinates
    layers: list[list[int]] = []  # Each layer contains 0-indexed atom indices
    layer_positions: list[float] = []  # Z-position of each layer

    current_layer = [sorted_indices[0]]
    current_z = sorted_coords[0]

    for i in range(1, len(sorted_coords)):
        z = sorted_coords[i]
        idx = sorted_indices[i]

        if abs(z - current_z) <= layer_tolerance:
            # Same layer
            current_layer.append(idx)
        else:
            # New layer - save previous
            layers.append(current_layer)
            layer_positions.append(np.mean([cart_coords_vac[j] for j in current_layer]))
            current_layer = [idx]
            current_z = z

    # Add final layer
    if current_layer:
        layers.append(current_layer)
        layer_positions.append(np.mean([cart_coords_vac[j] for j in current_layer]))

    n_layers = len(layers)

    # Identify top and bottom surface layers
    # Top surface = highest z layers (last in sorted list)
    # Bottom surface = lowest z layers (first in sorted list)

    top_surface_indices: list[int] = []
    bottom_surface_indices: list[int] = []
    bulk_indices: list[int] = []

    for layer_idx, layer_atoms in enumerate(layers):
        is_top = layer_idx >= n_layers - surface_layers
        is_bottom = layer_idx < surface_layers

        for atom_idx in layer_atoms:
            atom_1indexed = atom_idx + 1  # Convert to 1-indexed

            if is_top:
                top_surface_indices.append(atom_1indexed)
            elif is_bottom:
                bottom_surface_indices.append(atom_1indexed)
            else:
                bulk_indices.append(atom_1indexed)

    # Combine surface atoms based on settings
    if include_both_surfaces:
        surface_indices = top_surface_indices + bottom_surface_indices
    else:
        surface_indices = top_surface_indices

    return {
        "surface": sorted(surface_indices),
        "bulk": sorted(bulk_indices),
        "top_surface": sorted(top_surface_indices),
        "bottom_surface": sorted(bottom_surface_indices),
        "vacuum_direction": vacuum_direction,
        "n_layers": n_layers,
        "layer_positions": layer_positions,
    }


def apply_diffuse_basis_to_surface(
    structure: Structure,
    surface_basis: str = "TZP",
    bulk_basis: str = "DZP",
    surface_layers: int = 1,
    layer_tolerance: float = 0.5,
    vacuum_direction: str | None = None,
    include_both_surfaces: bool = True,
) -> tuple[list[str], dict[str, str], dict]:
    """
    Automatically apply larger (more diffuse) basis sets to surface atoms.

    This is a convenience function that combines surface atom detection
    with basis set application. Surface atoms get a larger basis set
    (more diffuse orbitals) to better describe the electronic structure
    at the vacuum interface.

    Parameters
    ----------
    structure : Structure
        Pymatgen structure (should be a slab with vacuum)
    surface_basis : str
        Basis size for surface atoms. Default: "TZP" (triple-zeta polarized)
    bulk_basis : str
        Basis size for bulk atoms. Default: "DZP" (double-zeta polarized)
    surface_layers : int
        Number of outermost atomic layers to treat as "surface".
        Default: 1 (only the layer directly at the vacuum interface)
    layer_tolerance : float
        Tolerance for grouping atoms into layers (Angstroms).
        Default: 0.5 Å
    vacuum_direction : str, optional
        Direction of vacuum: 'a', 'b', or 'c'. If None, auto-detected.
    include_both_surfaces : bool
        If True, apply to both surfaces. Default: True

    Returns
    -------
    species_labels : list[str]
        Species label for each atom
    pao_basissizes : dict[str, str]
        Dictionary for %block PAO.BasisSizes
    detection_info : dict
        Information about detected surface/bulk atoms

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.jobs.core import RelaxMaker
    >>>
    >>> # Load slab
    >>> slab = Structure.from_file("Pt111_slab.cif")
    >>>
    >>> # Apply diffuse basis to surface atoms (1 outermost layer)
    >>> species_labels, pao_basissizes, info = apply_diffuse_basis_to_surface(
    ...     slab,
    ...     surface_basis="TZP",  # Diffuse for surface
    ...     bulk_basis="DZP",  # Standard for bulk
    ...     surface_layers=1,  # Only outermost layer
    ... )
    >>>
    >>> print(f"Surface atoms ({len(info['surface'])}): {info['surface']}")
    >>> print(f"Bulk atoms ({len(info['bulk'])}): {info['bulk']}")
    >>> print(f"PAO.BasisSizes: {pao_basissizes}")
    >>>
    >>> # Add species labels to structure
    >>> slab.add_site_property("species_label", species_labels)
    >>>
    >>> # Create maker with the basis sizes
    >>> maker = RelaxMaker.fixed_cell_relaxation(
    ...     user_params={
    ...         "%block PAO.BasisSizes": pao_basissizes,
    ...     }
    ... )
    >>> job = maker.make(slab)

    Notes
    -----
    Why diffuse orbitals for surfaces?
    ----------------------------------
    At surfaces, electrons extend further into the vacuum than in bulk.
    Standard basis sets (optimized for bulk) may not describe this well.
    Using larger cutoff radii (more diffuse orbitals) at surface atoms
    improves:
    - Surface energies
    - Work functions
    - Adsorption energies
    - Surface band structures

    Typical settings:
    -----------------
    - Metals: surface_layers=1, TZP surface, DZP bulk
    - Oxides: surface_layers=1-2, TZP surface, DZP bulk
    - 2D materials: surface_layers=1 (just the layer), TZDP surface
    """
    # Detect surface atoms
    detection_info = detect_surface_atoms(
        structure,
        surface_layers=surface_layers,
        layer_tolerance=layer_tolerance,
        vacuum_direction=vacuum_direction,
        include_both_surfaces=include_both_surfaces,
    )

    # Create per-atom basis dict
    per_atom_basis: dict[int, str] = {}

    # Apply surface basis to surface atoms
    for atom_idx in detection_info["surface"]:
        per_atom_basis[atom_idx] = surface_basis

    # Apply bulk basis to bulk atoms (or use fallback)
    for atom_idx in detection_info["bulk"]:
        per_atom_basis[atom_idx] = bulk_basis

    # Generate species labels and PAO.BasisSizes
    species_labels, pao_basissizes = apply_per_atom_basis(
        structure, per_atom_basis, fallback_basis=bulk_basis
    )

    return species_labels, pao_basissizes, detection_info
