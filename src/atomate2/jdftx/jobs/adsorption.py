"""Core jobs for running JDFTx calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import job

from atomate2.jdftx.jobs.base import BaseJdftxMaker
from atomate2.jdftx.sets.core import (
    IonicMinSetGenerator,
)

if TYPE_CHECKING:
    from atomate2.jdftx.sets.core import JdftxInputGenerator
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)

@dataclass
class SurfaceMinMaker(BaseJdftxMaker):
    """Maker to create surface relaxation job."""
    name: str = "surface_ionic_min"
    input_set_generator: JdftxInputGenerator = field(
        default_factory=IonicMinSetGenerator(
            coulomb_truncation = True,
            auto_kpoint_density = 1000,
            calc_type="surface",
        )
    )

class MolMinMaker(BaseJdftxMaker):
    """Maker to create molecule relaxation job."""
    name: str = "surface_ionic_min"
    input_set_generator: JdftxInputGenerator = field(
        default_factory=IonicMinSetGenerator(
            coulomb_truncation = True,
            calc_type="molecule",
        )
    )



@job
def generate_slab(
    bulk_structure: Structure,
    min_slab_size: float,
    surface_idx: tuple,
    min_vacuum_size: float,
    min_lw: float,
) -> Structure:
    """Generate the adsorption slabs without adsorbates.
        Copied from 

    Parameters
    ----------
    bulk_structure: Structure
        The bulk/unit cell structure.
    min_slab_size: float
        The minimum size of the slab. In Angstroms or number of hkl planes.
        See pymatgen.core.surface.SlabGenerator for more details.
    surface_idx: tuple
        The Miller index [h, k, l] of the surface.
    min_vacuum_size: float
        The minimum size of the vacuum region. In Angstroms or number of hkl planes.
        See pymatgen.core.surface.SlabGenerator for more details.
    min_lw: float
        The minimum length and width of the slab.

    Returns
    -------
    Structure
        The slab structure without adsorbates.
    """