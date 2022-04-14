"""Definition of defect job maker."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from jobflow import Flow, Maker, Response, job
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar

from atomate2.common.files import copy_files, gunzip_files, gzip_files, rename_files
from atomate2.utils.file_client import FileClient
from atomate2.utils.path import strip_hostname
from atomate2.vasp.files import copy_vasp_outputs
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.run import run_vasp
from atomate2.vasp.schemas.defect import (
    CCDDocument,
    FiniteDifferenceDocument,
    WSWQDocument,
)
from atomate2.vasp.schemas.task import TaskDocument

logger = logging.getLogger(__name__)


@job
def spawn_energy_curve_calcs(
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
        suffix = f" {i+1}" if add_name == "" else f" {add_name} {i}"
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


@dataclass
class FiniteDifferenceMaker(Maker):
    """
    A maker to print and store WSWQ files.

    Reads the WAVECAR file and computs the desired quantities.
    This can be used in cases where data from the same calculation is used multiple times.

    Since all of the standard outputs are presumably already stored in the database,
    the make function here should only only store new data.
    """

    name: str = "finite diff"
    run_vasp_kwargs: dict = field(default_factory=dict)

    @job(data=WSWQDocument, output_schema=FiniteDifferenceDocument)
    def make(self, ref_calc_dir: str, distorted_calc_dirs: List[str]):
        """Run a post-processing VASP job."""
        fc = FileClient()
        copy_vasp_outputs(
            ref_calc_dir, additional_vasp_files=["WAVECAR"], file_client=fc
        )

        """Update the INCAR."""
        incar = Incar.from_file("INCAR")
        incar.update({"ALGO": "None", "NSW": 0, "LWAVE": False, "LWSWQ": True})
        incar.write_file("INCAR")

        d_dir_names = [strip_hostname(d) for d in distorted_calc_dirs]

        gunzip_files(
            allow_missing=True,
            force=True,
            include_files=["INCAR", "POSCAR", "WAVECAR", "POTCAR", "KPOINTS"],
        )
        for i, dir_name in enumerate(d_dir_names):
            # Copy a distorted WAVECAR to WAVECAR.qqq
            copy_files(dir_name, include_files=["WAVECAR.gz"], prefix="qqq.")
            gunzip_files(include_files="qqq.WAVECAR*", allow_missing=True)
            rename_files({"qqq.WAVECAR": "WAVECAR.qqq"})

            run_vasp(**self.run_vasp_kwargs)
            fc.copy(Path("WSWQ"), f"WSWQ.{i}")

        cur_dir = Path.cwd()
        fd_doc = FiniteDifferenceDocument.from_directory(cur_dir)
        gzip_files(cur_dir, force=True)
        return fd_doc
