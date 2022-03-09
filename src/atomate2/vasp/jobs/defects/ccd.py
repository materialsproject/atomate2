"""Jobs for performing wavefunction overlap calculations in VASP."""

from __future__ import annotations

import logging

import numpy as np
from jobflow import Flow, Response, job
from pymatgen.core import Structure

from atomate2.vasp.jobs.core import StaticMaker

logger = logging.getLogger(__name__)


@job
def run_ccd_cal(
    ref: Structure,
    distorted: Structure,
    static_maker: StaticMaker,
    distortions=(-0.1, 0.05, 0.05, 0.1, 1),
):
    """
    Compute the Configuration-coordinate diagram along the distortion direction between two structure.

    Parameters
    ----------
    ref : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the ground (final) state
    distorted : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the excited (initial) state
    static_maker : atomate2.vasp.jobs.core.StaticMaker
        StaticMaker object
    distortions : tuple
        list of distortions to apply

    Returns
    -------
    Response
        Response object
    """
    distorted_structures = ref.interpolate(distorted, nimages=sorted(distortions))
    uuids = []
    job_dirs = []
    jobs = []
    outputs = []

    # add the static job for the reference structure
    static_maker.make(ref)
    jobs.append(static_maker.make(ref))

    # add all the distorted structures
    for _, d_struct in enumerate(distorted_structures):
        static_job = static_maker.make(d_struct)
        jobs.append(static_job)
        static_job.uuid
        # dQ = get_dQ(ref, d_struct)
        outputs.append(static_job.output)
        uuids.append(static_job.output.uuid)
        job_dirs.append(static_job.output.dir_name)

    # TODO should construct some kind of CCD document class to deal with this
    flow_ = Flow(jobs, outputs)
    return Response(replace=flow_)
    return outputs


def get_dQ(ref: Structure, distorted: Structure) -> float:
    """
    Calculate dQ from the initial and final structures.

    Parameters
    ----------
    ground : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the ground (final) state
    excited : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the excited (initial) state

    Returns
    -------
    float
        the dQ value (amu^{1/2} Angstrom)
    """
    return np.sqrt(
        np.sum(
            list(
                map(
                    lambda x: x[0].distance(x[1]) ** 2 * x[0].specie.atomic_mass,
                    zip(ref, distorted),
                )
            )
        )
    )
