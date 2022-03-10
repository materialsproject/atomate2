"""Definition of defect job maker."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from jobflow import Flow, Response, job
from pymatgen.core import Structure

from atomate2.vasp.jobs.core import StaticMaker

logger = logging.getLogger(__name__)


@job
def calculate_energy_curve(
    ref: Structure,
    distorted: Structure,
    distortions: Iterable[float],
    static_maker: StaticMaker,
    prev_vasp_dir: str | Path | None = None,
):
    """
    Compute the total energy curve as you distort a reference structure to a distorted structure.

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
    jobs = []
    outputs = []

    # add the static job for the reference structure
    static_maker.make(ref)

    distorted_structures = ref.interpolate(distorted, nimages=sorted(distortions))
    # add all the distorted structures
    for _, d_struct in enumerate(distorted_structures):
        static_job = static_maker.make(d_struct, prev_vasp_dir=prev_vasp_dir)
        jobs.append(static_job)
        outputs.append(static_job.output)

    add_flow = Flow(jobs, outputs)
    return Response(output=outputs, replace=add_flow)
