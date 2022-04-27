"""Definition of defect job maker."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from jobflow import Flow, Maker, Response, job
from numpy.typing import NDArray
from pydantic import BaseModel
from pymatgen.analysis.defect.core import Defect
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar
from pymatgen.io.vasp.outputs import WSWQ

from atomate2.common.files import copy_files, gunzip_files, gzip_files, rename_files
from atomate2.utils.file_client import FileClient
from atomate2.utils.path import strip_hostname
from atomate2.vasp.files import copy_vasp_outputs
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.run import run_vasp
from atomate2.vasp.schemas.defect import CCDDocument, FiniteDifferenceDocument
from atomate2.vasp.schemas.task import TaskDocument
from atomate2.vasp.sets.defect import AtomicRelaxSetGenerator

logger = logging.getLogger(__name__)

################################################################################
# Default settings                                                            ##
################################################################################

DEFAULT_DISTORTIONS = (-1, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 1)
DEFECT_INCAR_SETTINGS = {
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "KSPACING": None,
    "ENCUT": 500,
    "LVHAR": True,
}
DEFECT_KPOINT_SETTINGS = {"reciprocal_density": 64}

DEFECT_RELAX_GENERATOR: AtomicRelaxSetGenerator = AtomicRelaxSetGenerator(
    use_structure_charge=True,
    user_incar_settings=DEFECT_INCAR_SETTINGS,
    user_kpoints_settings=DEFECT_KPOINT_SETTINGS,
)

DEFAULT_RELAX_MAKER = RelaxMaker(input_set_generator=DEFECT_RELAX_GENERATOR)
DEFAULT_RELAX_MAKER.input_set_generator.user_incar_settings.update({"LVHAR": True})


################################################################################
# Formation Energy                                                          ####
################################################################################


@job
def perform_defect_calculations(
    defect: Defect,
    relax_maker: RelaxMaker | None = None,
    prev_vasp_dir: str | Path | None = None,
    sc_mat: NDArray | None = None,
    add_name: str = "",
):
    """Perform charge defect supercell calculations and save the Hartree potential.

    Parameters
    ----------
    defect
        A defect object representing the defect in a unit cell.
    relax_maker
        A RelaxMaker object to use for the atomic relaxation. If None, the default will be used (see DEFAULT_RELAX_MAKER).
    prev_vasp_dir
        The directory containing the previous VASP calculation.
    sc_mat
        The supercell matrix. If None, the code will attempt to create a nearly-cubic supercell.
    add_name
        A string to add to the end of the calculation name.
    """
    jobs = []
    outputs = []
    sc_def_struct = defect.get_supercell_structure(sc_mat=sc_mat)
    relax_maker = relax_maker or DEFAULT_RELAX_MAKER
    for i in defect.get_charge_states():
        suffix = f" q={i}" if add_name == "" else f" {add_name} q={i}"
        charged_struct = sc_def_struct.copy()
        charged_struct.set_charge(i)
        charged_relax = relax_maker.make(charged_struct, prev_vasp_dir=prev_vasp_dir)
        charged_relax.append_name(suffix)
        jobs.append(charged_relax)
        charge_output: TaskDocument = charged_relax.output
        outputs.append(
            {
                "structure": charge_output.structure,
                "energy": charge_output.output.energy,
                "dir_name": charge_output.dir_name,
                "uuid": charged_relax.uuid,
            }
        )
    add_flow = Flow(jobs, outputs)
    return Response(output=outputs, replace=add_flow)


################################################################################
# Configuration-Coordinate-Diagram (CCD)                                    ####
################################################################################
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
    """Compute the total energy curve as you distort a reference structure to a distorted structure.

    Parameters
    ----------
    relaxed_structure : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the ground (final) state
    distorted_structure : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the excited (initial) state
    static_maker : atomate2.vasp.jobs.core.StaticMaker
        StaticMaker object
    distortions : tuple
        list of distortions to apply
    add_name : str
        additional name to add to the flow name
    add_info : dict
        additional info to add to the to a info.json file for each static calculation.
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

        static_job.append_name(suffix)
        jobs.append(static_job)

        # only pass the information needed by CCDInput to the next job
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
        List of CCDInput objects
    inputs2 : Iterable[CCDInput]
        List of CCDInput objects
    undistorted_index : int
        Index of the undistorted structure in the list of distorted structures

    Returns
    -------
    Response
        Response object
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


@dataclass
class FiniteDifferenceMaker(Maker):
    """
    A maker to print and store WSWQ files.

    Reads the WAVECAR file and computs the desired quantities.
    This can be used in cases where data from the same calculation is used multiple times.

    Since all of the standard outputs are presumably already stored in the database,
    the make function here should only only store new data.

    Parameters
    ----------
    name : str
        Name of the jobs created by this maker
    run_vasp_kwargs : dict
        kwargs to pass to run_vasp
    """

    name: str = "finite diff"
    run_vasp_kwargs: dict = field(default_factory=dict)

    @job(data=WSWQ, output_schema=FiniteDifferenceDocument)
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
