"""Definition of defect job maker."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from jobflow import Flow, Response, job
from pymatgen.core import Structure

from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.schemas.defect import CCDDocument
from atomate2.vasp.schemas.task import TaskDocument

logger = logging.getLogger(__name__)


@job
def calculate_energy_curve(
    ref: Structure,
    distorted: Structure,
    distortions: Iterable[float],
    static_maker: StaticMaker,
    prev_vasp_dir: str | Path | None = None,
    add_name: str = "",
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
    for i, d_struct in enumerate(distorted_structures):
        static_job = static_maker.make(d_struct, prev_vasp_dir=prev_vasp_dir)
        suffix = f"{i+1}" if add_name == "" else f" {add_name} {i}"
        static_job.append_name(f"{suffix}")
        jobs.append(static_job)
        outputs.append(static_job.output)

    add_flow = Flow(jobs, outputs)
    return Response(output=outputs, replace=add_flow)


@job(output_schema=CCDDocument)
def get_ccd_from_task_docs(
    taskdocs1: Iterable[TaskDocument],
    taskdocs2: Iterable[TaskDocument],
    structure1: Structure,
    structure2: Structure,
):
    """
    Get the configuration coordinate diagram from the task documents.

    Parameters
    ----------
    taskdocs1 : Iterable[TaskDocument]
        task documents for the first charge state
    taskdocs2 : Iterable[TaskDocument]
        task documents for the second charge state
    structure1 : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the ground (final) state
    structure2 : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the excited (initial) state

    Returns
    -------
    Response
        Response object
    """
    ccd_doc = CCDDocument.from_distorted_calcs(
        taskdocs1,
        taskdocs2,
        structure1=structure1,
        structure2=structure2,
    )
    return Response(output=ccd_doc)
