"""Jobs for running qha calculations."""

from __future__ import annotations

import contextlib
import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Response, job
from phonopy import Phonopy
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from atomate2.common.schemas.phonons import ForceConstants, PhononBSDOSDoc, get_factor

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.forcefields.jobs import ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker


logger = logging.getLogger(__name__)


@job
def get_phonon_jobs(phonon_maker, output: dict
) -> Flow:
    """
    Job that computes total DFT energy of the cell.

    Parameters
    ----------
    total_dft_energy_per_formula_unit: float
        Total DFT energy in eV per formula unit.
    structure: Structure object
        Corresponding structure object.
    """
    phonon_jobs=[]
    outputs=[]
    for structure in output["relax"]["structure"]:
        phonon_job=phonon_maker.make(structure)
        phonon_jobs.append(phonon_job)
        outputs.append(phonon_job.output)

    return Response(replace=phonon_jobs, output=outputs)



@job
def analyze_free_energy(eos_outputs, phonon_outputs
) -> Flow:
    """
    Job that analyzes the free energy from all phonon runs

    Parameters
    ----------
    total_dft_energy_per_formula_unit: float
        Total DFT energy in eV per formula unit.
    structure: Structure object
        Corresponding structure object.
    """
    # only add free energies if there are no imaginary modes
    # tolerance has to be tested
    free_energies={}

    for itemp, temp in enumerate(phonon_outputs[0].output.temperature_range):
        free_energies[temp]=[]
        for energy, output in zip(eos_outputs["relax"]["energies"],phonon_outputs):
            # convert all units to eV
            # correct this - currently wrong
            # how are energies normalized? per molecule?
            free_energies[temp].append(energy+output["free_energy"][itemp])

    print(free_energies)
    return free_energies


    # TODO: should return some output doc
    # have to think about how it should look like
    # need to check