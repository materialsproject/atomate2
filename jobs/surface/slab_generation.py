"""Jobs for slab generation integrated into workflows."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from jobflow import job
from pymatgen.core import Structure
from pymatgen.core.surface import (
    SlabGenerator,
    get_symmetrically_distinct_miller_indices,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@job
def generate_slabs_for_miller_index(
    structure: Structure,
    miller_index: tuple[int, int, int],
    min_slab_size: float = 10.0,
    min_vacuum_size: float = 15.0,
    max_index: int = 1,
    symmetrize: bool = False,
    lll_reduce: bool = False,
    center_slab: bool = True,
    primitive: bool = True,
    max_normal_search: int = 1,
) -> dict[str, Any]:
    """
    Generate all unique slab terminations for a given Miller index.

    Uses pymatgen's SlabGenerator to create slabs.

    Parameters
    ----------
    structure : Structure
        Bulk structure.
    miller_index : tuple[int, int, int]
        Miller indices (h, k, l).
    min_slab_size : float
        Minimum slab thickness in Ångströms.
    min_vacuum_size : float
        Minimum vacuum thickness in Ångströms.
    max_index : int
        Maximum Miller index for symmetry analysis.
    symmetrize : bool
        Whether to create symmetric slabs.
    lll_reduce : bool
        Whether to perform LLL reduction.
    center_slab : bool
        Whether to center the slab in the cell.
    primitive : bool
        Whether to use primitive cell.
    max_normal_search : int
        Maximum search range for surface normal.

    Returns
    -------
    dict
        Dictionary containing:
        - 'miller_index': tuple
        - 'slabs': list of Structure objects
        - 'n_terminations': int
        - 'termination_labels': list of str
        - 'metadata': dict with slab generation info
    """
    logger.info(f"generate_slabs_for_miller_index: {miller_index}")

    miller_h, miller_k, miller_l = miller_index

    # Create SlabGenerator
    slabgen = SlabGenerator(
        initial_structure=structure,
        miller_index=miller_index,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        lll_reduce=lll_reduce,
        center_slab=center_slab,
        primitive=primitive,
        max_normal_search=max_normal_search,
    )

    # Get all unique slabs
    slabs = slabgen.get_slabs(
        bonds=None,  # Auto-detect bonds
        ftol=0.1,  # Fractional tolerance for layer identification
        tol=0.1,  # Distance tolerance in Angstroms
        max_broken_bonds=0,  # Prefer unbroken bonds
        symmetrize=symmetrize,
    )

    logger.info(f"Generated {len(slabs)} slab(s) for ({miller_h}{miller_k}{miller_l})")

    # Create termination labels
    termination_labels = []
    for i, slab in enumerate(slabs):
        # Get surface composition
        positions = slab.cart_coords
        z_coords = positions[:, 2]
        z_max = z_coords.max()

        # Find top layer atoms (within 0.5 Å of max z)
        top_layer_indices = np.where(z_coords > z_max - 0.5)[0]
        top_species = [slab.species[idx].symbol for idx in top_layer_indices]

        # Create label from most common species
        from collections import Counter

        species_count = Counter(top_species)
        dominant_species = species_count.most_common(1)[0][0]

        # Create unique label
        label = f"{dominant_species}_term{i + 1}"
        termination_labels.append(label)

        logger.info(f"  Termination {i + 1}: {label} ({len(slab)} atoms)")

    # Collect metadata
    metadata = {
        "bulk_formula": structure.composition.reduced_formula,
        "bulk_n_atoms": len(structure),
        "min_slab_size": min_slab_size,
        "min_vacuum_size": min_vacuum_size,
        "symmetrize": symmetrize,
        "primitive": primitive,
    }

    return {
        "miller_index": miller_index,
        "slabs": slabs,
        "n_terminations": len(slabs),
        "termination_labels": termination_labels,
        "metadata": metadata,
    }


@job
def generate_slabs_for_all_miller_indices(
    structure: Structure,
    miller_indices: list[tuple[int, int, int]] | None = None,
    max_index: int = 1,
    min_slab_size: float = 10.0,
    min_vacuum_size: float = 15.0,
    symmetrize: bool = False,
) -> dict[str, Any]:
    """
    Generate slabs for multiple Miller indices.

    Parameters
    ----------
    structure : Structure
        Bulk structure.
    miller_indices : list[tuple], optional
        List of Miller indices. If None, generates all unique indices up to max_index.
    max_index : int
        Maximum Miller index to consider if miller_indices is None.
    min_slab_size : float
        Minimum slab thickness in Ångströms.
    min_vacuum_size : float
        Minimum vacuum thickness in Ångströms.
    symmetrize : bool
        Whether to create symmetric slabs.

    Returns
    -------
    dict
        Dictionary mapping Miller indices to slab data.
    """
    logger.info("generate_slabs_for_all_miller_indices.__init__()")

    if miller_indices is None:
        # Get all symmetrically distinct Miller indices
        miller_indices = get_symmetrically_distinct_miller_indices(structure, max_index)
        logger.info(
            f"Auto-generated {len(miller_indices)} unique Miller indices up to {max_index}"
        )
    else:
        logger.info(f"Using {len(miller_indices)} user-specified Miller indices")

    # Log all Miller indices
    for hkl in miller_indices:
        logger.info(f"  ({hkl[0]} {hkl[1]} {hkl[2]})")

    # Generate slabs for each Miller index
    all_slab_data = {}

    for hkl in miller_indices:
        slab_data = generate_slabs_for_miller_index(
            structure=structure,
            miller_index=hkl,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            max_index=max_index,
            symmetrize=symmetrize,
        )

        all_slab_data[str(hkl)] = slab_data

    return {
        "bulk_structure": structure,
        "miller_indices": miller_indices,
        "slab_data": all_slab_data,
        "n_miller_indices": len(miller_indices),
    }
