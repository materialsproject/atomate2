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
    from emmet.core.math import Matrix3D

    from atomate2.forcefields.jobs import ForceFieldStaticMaker
    from atomate2.vasp.sets.base import VaspInputGenerator

logger = logging.getLogger(__name__)


@job
def get_molecule_structure(molecule: Molecule) -> Structure:
    """Get the molecule structure."""
    return molecule.get_boxed_structure(10, 10, 10, offset=np.array([5, 5, 5]))

@job
def generate_slab_job(
        structure: Structure,
        supercell_index,
        surface_idx,
        prefer_90_degrees,
        min_vacuum
        ):


@job
def generate_adslab_jobs(
    bulk_structure,
    molecule_structure,
    min_slab_size,
    surface_idx,
    min_vacuum_size,
    min_lw,
    ):

    slab_generator = SlabGenerator(bulk_structure, surface_idx, min_slab_size, min_vacuum_size, primitive=False, center_slab=True)
    slab = slab_generator.get_slab()
    ads_slabs = AdsorbateSiteFinder(slab).generate_adsorption_structures(molecule_structure, translate=True, min_lw)

@job
def run_adsorption_calculations():
    """Generate adsorption structures."""
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
    name: str = "adsorption static calculation
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