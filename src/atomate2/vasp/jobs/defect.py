"""Definition of defect job maker."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import numpy as np
from jobflow import Flow, Maker, Response, job
from numpy.typing import NDArray
from pydantic import BaseModel
from pymatgen.analysis.defect.core import Defect
from pymatgen.analysis.defect.generators import DefectGenerator
from pymatgen.analysis.defect.supercells import (
    get_matched_structure_mapping,
    get_sc_fromstruct,
)
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
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

_logger = logging.getLogger(__name__)

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


class BulkSuperCellSummary(BaseModel):
    """Document model containing information about the bulk supercell calculation."""

    uc_structure: Structure
    sc_entry: ComputedStructureEntry
    sc_mat: List[List[float]]
    dir_name: str
    uuid: str


@job(
    name="bulk supercell",
)
def bulk_supercell_calculation(
    uc_structure: Structure,
    relax_maker: RelaxMaker,
    sc_mat: NDArray | None = None,
    bulk_info: BulkSuperCellSummary | None = None,
) -> BulkSuperCellSummary:
    """Bulk Supercell calculation.

    Check if the information from a bulk supercell calculation has been provided.
    If not, run a bulk supercell calculation.

    Parameters
    ----------
    uc_structure : Structure
        The unit cell structure.
    relax_maker : RelaxMaker
        The relax maker to use.
    sc_mat : NDArray | None
        The supercell matrix. if None, the code will attempt to create a nearly-cubic supercell.
    bulk_info : BulkSuperCellSummary | None
        The bulk supercell calculation summary. If information from a previous calculation is provided
        the bulk supercell calculation will be skipped.

    Returns
    -------
    dict:
        The bulk supercell calculation summary.
    """
    if bulk_info is not None:
        sc_structure = bulk_info.sc_entry.structure
        (map_sc,) = get_matched_structure_mapping(
            uc_structure, sc_structure
        )  # TODO: this might need relaxing if too tight
        summary_d = bulk_info.dict()
        summary_d["sc_mat"] = map_sc.tolist()
        summary = summary_d
        return Response(output=summary)
    else:
        _logger.info("Bulk supercell calculation not found. Running...")
        sc_mat = get_sc_fromstruct(uc_structure) if sc_mat is None else sc_mat
        sc_mat = np.array(sc_mat)
        sc_structure = uc_structure * sc_mat
        relax_job = relax_maker.make(sc_structure)
        relax_job.name = "bulk relax"
        relax_output: TaskDocument = relax_job.output

        summary_d = dict(
            uc_structure=uc_structure,
            sc_entry=relax_output.entry,
            sc_struct=relax_output.structure,
            sc_mat=sc_mat.tolist(),
            dir_name=relax_output.dir_name,
            uuid=relax_job.uuid,
        )

        summary_job = create_summary(summary_d)  # This feels a little awkward
        return Response(output=summary_job.output, replace=[relax_job, summary_job])


@job(output_schema=BulkSuperCellSummary)
def create_summary(d: dict) -> BulkSuperCellSummary:
    """Create a summary from a bulk supercell calculation."""
    entry = d["sc_entry"]
    structure = d["sc_struct"]
    entry_d = entry.as_dict()
    entry_d["structure"] = structure.as_dict()
    d["sc_entry"] = ComputedStructureEntry.from_dict(entry_d)
    return BulkSuperCellSummary(**d)


@job
def spawn_defects_calcs(
    defect_gen: DefectGenerator, sc_mat: NDArray, bulk_summary: BulkSuperCellSummary
) -> Response:
    """Spawn defect calculations from the DefectGenerator.

    Parameters
    ----------
    defect_gen : DefectGenerator
        The defect generator to use.
    sc_mat : NDArray
        The supercell matrix. If None, the code will attempt to create a nearly-cubic supercell.
    bulk_summary : BulkSuperCellSummary
        The bulk supercell calculation summary.

    Returns
    -------
    Response:
        The response containing the outputs of the defect calculations as a dictionary
    """
    bulk_summary.sc_entry

    defect_calcs = []
    output = dict(bulk_sc_entry=bulk_summary.sc_entry, defect_calcs=dict())
    name_counter: dict = defaultdict(lambda: 0)

    for defect in defect_gen:
        defect_job = perform_defect_calcs(
            defect,
            sc_mat=sc_mat,
            prev_vasp_dir=bulk_summary.dir_name,
            defect_index=f"{name_counter[defect.name]}",
        )
        defect_calcs.append(defect_job)
        output["defect_calcs"][f"{defect.name}-{name_counter[defect.name]}"] = dict(
            defect=defect,
            results=defect_job.output,
        )
        name_counter[defect.name] += 1

    return Response(output=output, replace=defect_calcs)


@job
def perform_defect_calcs(
    defect: Defect,
    relax_maker: RelaxMaker | None = None,
    prev_vasp_dir: str | Path | None = None,
    sc_mat: NDArray | None = None,
    defect_index: str = "",
) -> Response:
    """Perform charge defect supercell calculations and save the Hartree potential.

    Parameters
    ----------
    defect23
        A defect object representing the defect in a unit cell.
    relax_maker
        A RelaxMaker object to use for the atomic relaxation. If None, the default will be used (see DEFAULT_RELAX_MAKER).
    prev_vasp_dir
        The directory containing the previous VASP calculation.
    sc_mat
        The supercell matrix. If None, the code will attempt to create a nearly-cubic supercell.
    defect_index
        Additional index to give unique names to the defect calculations.

    Returns
    -------
    Response
        A response object containing the summary of the calculations for different charge states.

    """
    jobs = []
    outputs = dict()
    sc_def_struct = defect.get_supercell_structure(sc_mat=sc_mat)
    relax_maker = relax_maker or DEFAULT_RELAX_MAKER
    for qq in defect.get_charge_states():
        suffix = (
            f" {defect.name} q={qq}"
            if defect_index == ""
            else f" {defect.name}-{defect_index} q={qq}"
        )
        charged_struct = sc_def_struct.copy()
        charged_struct.set_charge(qq)
        charged_relax = relax_maker.make(charged_struct, prev_vasp_dir=prev_vasp_dir)
        charged_relax.append_name(suffix)
        jobs.append(charged_relax)
        charge_output: TaskDocument = charged_relax.output

        outputs[qq] = {
            "structure": charge_output.structure,
            "entry": charge_output.entry,
            "dir_name": charge_output.dir_name,
            "uuid": charged_relax.uuid,
        }

    add_flow = Flow(jobs, output=outputs)
    return Response(output=outputs, replace=add_flow)


@job
def collect_defect_outputs(spawn_defects_output: dict) -> dict:
    """Collect all the outputs from the defect calculations.

    This job will combine the structure and entry fields to create a ComputerStructureEntry object.
    """
    defect_calcs_output = spawn_defects_output["defect_calcs"]
    for k, v in defect_calcs_output.items():
        for qq, res in v["results"].items():
            structure = res.pop("structure")
            entry = res.pop("entry")
            entry_d = entry.as_dict()
            entry_d["structure"] = structure.as_dict()
            res["entry"] = ComputedStructureEntry.from_dict(entry_d)
    return spawn_defects_output


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
    distortions : Iterable[float]
        list of distortions, as a fraction of ΔQ, to apply
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

        for i, dir_name in enumerate(d_dir_names):
            # Copy a distorted WAVECAR to WAVECAR.qqq
            copy_files(dir_name, include_files=["WAVECAR.gz"], prefix="qqq.")
            gunzip_files(include_files="qqq.WAVECAR*", allow_missing=True)
            rename_files({"qqq.WAVECAR": "WAVECAR.qqq"})

            run_vasp(**self.run_vasp_kwargs)
            fc.copy("WSWQ", f"WSWQ.{i}")

        fd_doc = FiniteDifferenceDocument.from_directory(
            ".", ref_dir=ref_calc_dir, distorted_dirs=d_dir_names
        )
        gzip_files(".", force=True)
        return fd_doc
