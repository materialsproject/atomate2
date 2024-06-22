"""Jobs for defect calculations."""

from __future__ import annotations

import logging

from jobflow import job
from pymatgen.io.vasp import Incar
from pymatgen.io.vasp.outputs import WSWQ

from atomate2.common.files import copy_files, gunzip_files, gzip_files, rename_files
from atomate2.common.jobs.defect import (  # noqa: F401
    bulk_supercell_calculation,
    get_ccd_documents,
    get_charged_structures,
    get_supercell_from_prv_calc,
    spawn_defect_q_jobs,
    spawn_energy_curve_calcs,
)
from atomate2.utils.file_client import FileClient
from atomate2.utils.path import strip_hostname
from atomate2.vasp.files import copy_vasp_outputs
from atomate2.vasp.run import run_vasp
from atomate2.vasp.schemas.defect import FiniteDifferenceDocument

logger = logging.getLogger(__name__)


@job(data=WSWQ, output_schema=FiniteDifferenceDocument)
def calculate_finite_diff(
    distorted_calc_dirs: list[str],
    ref_calc_index: int,
    run_vasp_kwargs: dict | None = None,
) -> FiniteDifferenceDocument:
    """Run a post-processing VASP job for the finite difference overlap.

    Reads the WAVECAR file and computes the desired quantities. This can be used in
    cases where data from the same calculation is used multiple times.

    Since all of the standard outputs are presumably already stored in the database,
    the make function here should only store new data.

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
    run_vasp_kwargs = run_vasp_kwargs or {}
    fc = FileClient()
    copy_vasp_outputs(ref_calc_dir, additional_vasp_files=["WAVECAR"], file_client=fc)

    # Update the INCAR for the WSWQ calculation
    incar = Incar.from_file("INCAR")
    incar.update(ALGO="None", NSW=0, LWAVE=False, LWSWQ=True)
    incar.write_file("INCAR")

    d_dir_names = [strip_hostname(d) for d in distorted_calc_dirs]

    for idx, dir_name in enumerate(d_dir_names):
        # Copy a distorted WAVECAR to WAVECAR.qqq
        copy_files(dir_name, include_files=["WAVECAR.gz"], prefix="qqq.")
        gunzip_files(include_files="qqq.WAVECAR*", allow_missing=True)
        rename_files({"qqq.WAVECAR": "WAVECAR.qqq"})

        run_vasp(**run_vasp_kwargs)
        fc.copy("WSWQ", f"WSWQ.{idx}")

    fd_doc = FiniteDifferenceDocument.from_directory(
        ".", ref_dir=ref_calc_dir, distorted_dirs=d_dir_names
    )
    gzip_files(".", force=True)
    return fd_doc
