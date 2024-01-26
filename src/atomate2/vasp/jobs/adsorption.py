"""Jobs used in the calculation of surface adsorption energy."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job
from pymatgen.core import Structure, Molecule, Element

from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import SlabGenerator
from pymatgen.transformations.standard_transformations import RotationTransformation

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from pathlib import Path
    import numpy as np
    from atomate2.vasp.sets.base import VaspInputGenerator

logger = logging.getLogger(__name__)


@job
def get_boxed_molecule(molecule: Molecule) -> Structure:
    """Get the molecule structure."""
    return molecule.get_boxed_structure(10, 10, 10, offset=np.array([5, 5, 5]))


@job
def removeAdsorbate(slab):
    """
    Remove adsorbate from the given slab.

    Parameters:
    slab (Slab): A pymatgen Slab object, potentially with adsorbates.

    Returns:
    Slab: The modified slab with adsorbates removed.
    """
    adsorbate_indices = []
    for i, site in enumerate(slab):
        if site.properties.get('surface_properties') == 'adsorbate':
            adsorbate_indices.append(i)

    # Reverse the indices list to avoid index shifting after removing sites
    adsorbate_indices.reverse()

    # Remove the adsorbate sites
    for idx in adsorbate_indices:
        slab.remove_sites([idx])

    return slab

@job
def run_adslab_jobs(
    bulk_structure,
    molecule_structure,
    min_slab_size,
    surface_idx,
    min_vacuum_size,
    min_lw,
    include_slab=True,
    ) -> Flow:

    slab_generator = SlabGenerator(bulk_structure, surface_idx, min_slab_size, min_vacuum_size, primitive=False, center_slab=True)
    slab = slab_generator.get_slab()
    ads_slabs = AdsorbateSiteFinder(slab).generate_adsorption_structures(molecule_structure, translate=True, min_lw=min_lw)

    if include_slab:
        H = Molecule([Element("H")], [[0, 0, 0]])
        temp_slab = slab_generator.get_slab()
        ads_slabs = AdsorbateSiteFinder(temp_slab).generate_adsorption_structures(H, translate=True, min_lw=min_lw)
        pureSlab = removeAdsorbate(ads_slabs[0])


@job
def run_adsorption_calculations():
    """Calculating the adsorption energies."""
    pass

@dataclass
class moleculeRelaxMaker(BaseVaspMaker):
    name: str = "adsorption relaxation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"grid_density": 7_000},
            user_incar_settings={
                "ALGO": "Normal",
                "IBRION": 2,
                "ISIF": 2,
                "ENCUT": 700,
                "GGA": "RP",
                "EDIFF": 1e-5,
                "LAECHG": False,
                "LREAL": False,
                "LCHARG": False,
                "LDAU": False,
                "NSW": 300,
                "NELM": 500,
            },
            auto_ispin=True,
        )
    )

@dataclass
class adslabRelaxMaker(BaseVaspMaker):
    name: str = "adsorption relaxation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"grid_density": 7_000},
            user_incar_settings={
                "ALGO": "Normal",
                "IBRION": 2,
                "ISIF": 2,
                "ENCUT": 700,
                "GGA": "RP",
                "EDIFF": 1e-5,
                "LAECHG": False,
                "LREAL": False,
                "LCHARG": False,
                "LDAU": False,
                "NSW": 300,
                "NELM": 500,
            },
            auto_ispin=True,
        )
    )

@dataclass
class StaticMaker(BaseVaspMaker):
    name: str = "adsorption static calculation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"grid_density": 7_000},
            user_incar_settings={
                "ALGO": "Normal",
                "ENCUT": 700,
                "GGA": "RP",
                "EDIFF": 1e-7,
                "LAECHG": False,
                "LREAL": False,
                "LCHARG": False,
                "LDAU": False,
                "NELM": 500,
            },
            auto_ispin=True,
        )
    )