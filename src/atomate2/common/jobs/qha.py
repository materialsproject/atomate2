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
from atomate2.common.jobs.eos import PostProcessEosEnergy
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
def analyze_free_energy(phonon_outputs
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
    fit_jobs = []
    fit_outputs={}
    for itemp, temp in enumerate(phonon_outputs[0].temperatures):
        volume = []
        free_energies[temp]=[]
        stress=[]

        for output in phonon_outputs:
            # convert all units to eV, normalize per formula unit
            # check if imaginary modes
            if not output.has_imaginary_modes:
                free_energy_eV=output.free_energies[itemp]*1.036*(10**(-5))
                free_energies[temp].append((output.total_dft_energy+free_energy_eV)*output.formula_units)
                volume.append(output.volume_per_formula_unit*output.formula_units)
                stress.append(output.stress)

        fit_dict={"relax":{"energy": free_energies[temp],"volume":volume, "stress":stress}}

        fitjob = PostProcessEosEnergy().make(fit_dict)
        fit_outputs[temp]={"fit": fitjob.output, "raw": fit_dict}
        fit_jobs.append(fitjob)


    return Response(replace=fit_jobs, output=fit_outputs)


    # TODO: should return some output doc
    # have to think about how it should look like
    # need to check

@job
def get_qha_results(fit_output):
    qha_results={}
    temps={}
    min_vol = {}
    min_free_energy = {}
    for temp, fit in fit_output.items():


        for eos, result in fit["fit"]["relax"]["EOS"].items():
            if eos not in min_vol:
                temps[eos]=[]
                min_vol[eos] = []
                min_free_energy[eos] = []
            # to check if the fit was fine
            if "v0" in result:
                temps[eos].append(float(temp))
                min_vol[eos].append(result["v0"])
                min_free_energy[eos].append(result["e0"])

    print(temps)
    print(min_vol)
    print(min_free_energy)
    alpha={}
    for eos, ts in temps.items():
        if len(ts) > 4:
            # get thermal expansion
            if eos not in alpha:
                alpha[eos]=[0.0]

                # check possible implementations
                # here: finite differences
                for i in range(1, len(ts) - 1):
                    # compute by finite differences
                    dt = ts[i + 1] - ts[i - 1]
                    dv = min_vol[eos][i + 1] - min_vol[eos][i - 1]
                    alpha[eos].append(dv / dt / min_vol[eos][i])
            # get heat capacities


    # Todo: add a typical qha plot for the final folder
    # needs all eos results here as well
    qha_results={}
    for eos, ts in temps.items():
        qha_results[eos]={}
        qha_results[eos]["temps"]=ts
        qha_results[eos]["alpha"]=alpha[eos]
        qha_results[eos]["min_vol"]=min_vol[eos]
        qha_results[eos]["min_free_energy"]=min_free_energy[eos]

    print(qha_results)
    return qha_results

