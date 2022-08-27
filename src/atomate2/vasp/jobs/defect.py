"""Jobs for defect calculations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from jobflow import Flow, Response, job
from pydantic import BaseModel
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar
from pymatgen.io.vasp.outputs import WSWQ

from atomate2.common.files import copy_files, gunzip_files, gzip_files, rename_files
from atomate2.utils.file_client import FileClient
from atomate2.utils.path import strip_hostname
from atomate2.vasp.files import copy_vasp_outputs
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.run import run_vasp
from atomate2.vasp.schemas.defect import CCDDocument, FiniteDifferenceDocument
from atomate2.vasp.schemas.task import TaskDocument

logger = logging.getLogger(__name__)


class CCDInput(BaseModel):
    """Document model to help construct CCDDocument."""

    structure: Structure
    energy: float
    dir_name: str
    uuid: str


@job
def spawn_energy_curve_calcs(
    relaxed_structure: Structure,
    distorted_structure: Structure,
    distortions: Iterable[float],
    static_maker: StaticMaker,
    prev_vasp_dir: str | Path | None = None,
    add_name: str = "",
    add_info: dict | None = None,
):
    """Compute the total energy curve from a reference to distorted structure.

    Parameters
    ----------
    relaxed_structure : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the ground (final) state.
    distorted_structure : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the excited (initial) state.
    static_maker : atomate2.vasp.jobs.core.StaticMaker
        StaticMaker object.
    distortions : Iterable[float]
        List of distortions, as a fraction of ΔQ, to apply.
    add_name : str
        Additional name to add to the flow name.
    add_info : dict
        Additional info to add to the to a info.json file for each static calculation.
        This data can be used to reconstruct the provenance of the calculation.

    Returns
    -------
    Response
        Response object
    """
    jobs = []
    outputs = []

    # add the static job for the reference structure
    static_maker.make(relaxed_structure)
    s_distortions = sorted(distortions)
    distorted_structures = relaxed_structure.interpolate(
        distorted_structure, nimages=s_distortions
    )
    # add all the distorted structures
    for i, d_struct in enumerate(distorted_structures):
        static_job = static_maker.make(d_struct, prev_vasp_dir=prev_vasp_dir)
        suffix = f" {i}" if add_name == "" else f" {add_name} {i}"

        # write some provenances data in info.json file
        info = {
            "relaxed_structure": relaxed_structure,
            "distorted_structure": distorted_structure,
            "distortion": s_distortions[i],
        }
        if add_info is not None:
            info.update(add_info)
        static_job.update_maker_kwargs(
            {"_set": {"write_additional_data->info:json": info}}, dict_mod=True
        )

        static_job.append_name(f"{suffix}")
        jobs.append(static_job)
        # outputs.append(static_job.output)
        task_doc: TaskDocument = static_job.output
        outputs.append(
            dict(
                structure=task_doc.output.structure,
                energy=task_doc.output.energy,
                dir_name=task_doc.dir_name,
                uuid=task_doc.uuid,
            )
        )

    add_flow = Flow(jobs, outputs)
    return Response(output=outputs, replace=add_flow)


@job(output_schema=CCDDocument)
def get_ccd_documents(
    inputs1: Iterable[CCDInput],
    inputs2: Iterable[CCDInput],
    undistorted_index: int,
):
    """
    Get the configuration coordinate diagram from the task documents.

    Parameters
    ----------
    inputs1 : Iterable[CCDInput]
        List of CCDInput objects.
    inputs2 : Iterable[CCDInput]
        List of CCDInput objects.
    undistorted_index : int
        Index of the undistorted structure in the list of distorted structures.

    Returns
    -------
    Response
        Response object.
    """
    static_uuids1 = [i["uuid"] for i in inputs1]
    static_uuids2 = [i["uuid"] for i in inputs2]

    ccd_doc = CCDDocument.from_task_outputs(
        structures1=[i["structure"] for i in inputs1],
        structures2=[i["structure"] for i in inputs2],
        energies1=[i["energy"] for i in inputs1],
        energies2=[i["energy"] for i in inputs2],
        static_dirs1=[i["dir_name"] for i in inputs1],
        static_dirs2=[i["dir_name"] for i in inputs2],
        static_uuids1=static_uuids1,
        static_uuids2=static_uuids2,
        relaxed_uuid1=static_uuids1[undistorted_index],
        relaxed_uuid2=static_uuids2[undistorted_index],
    )

    return Response(output=ccd_doc)


@job(data=WSWQ, output_schema=FiniteDifferenceDocument)
def calculate_finite_diff(
    distorted_calc_dirs: list[str],
    ref_calc_index: int,
    run_vasp_kwargs: dict | None = None,
):
    """Run a post-processing VASP job for the finite difference overlap.

    Reads the WAVECAR file and computs the desired quantities. This can be used in
    cases where data from the same calculation is used multiple times.

    Since all of the standard outputs are presumably already stored in the database,
    the make function here should only only store new data.

    Parameters
    ----------
    distorted_calc_dirs: List[str]
        List of directories containing distorted calculations.
    ref_calc_index: int
        Index of the reference (distortion=0) calculation.
    run_vasp_kwargs : dict
        kwargs to pass to run_vasp (should be copied from the static maker used for
        previous calculations).
    """
    ref_calc_dir = distorted_calc_dirs[ref_calc_index]
    run_vasp_kwargs = dict() if run_vasp_kwargs is None else run_vasp_kwargs
    fc = FileClient()
    copy_vasp_outputs(ref_calc_dir, additional_vasp_files=["WAVECAR"], file_client=fc)

    # Update the INCAR for the WSWQ calculation
    incar = Incar.from_file("INCAR")
    incar.update({"ALGO": "None", "NSW": 0, "LWAVE": False, "LWSWQ": True})
    incar.write_file("INCAR")

    d_dir_names = [strip_hostname(d) for d in distorted_calc_dirs]

    for i, dir_name in enumerate(d_dir_names):
        # Copy a distorted WAVECAR to WAVECAR.qqq
        copy_files(dir_name, include_files=["WAVECAR.gz"], prefix="qqq.")
        gunzip_files(include_files="qqq.WAVECAR*", allow_missing=True)
        rename_files({"qqq.WAVECAR": "WAVECAR.qqq"})

        run_vasp(**run_vasp_kwargs)
        fc.copy("WSWQ", f"WSWQ.{i}")

    fd_doc = FiniteDifferenceDocument.from_directory(
        ".", ref_dir=ref_calc_dir, distorted_dirs=d_dir_names
    )
    gzip_files(".", force=True)
    return fd_doc
