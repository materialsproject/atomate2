"""Jobs for defect calculations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from jobflow import Flow, Response, job
from numpy.typing import NDArray
from pydantic import BaseModel
from pymatgen.analysis.defects.core import Defect
from pymatgen.analysis.defects.supercells import (
    get_matched_structure_mapping,
    get_sc_fromstruct,
)
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.outputs import Vasprun

from atomate2.common.files import get_zfile
from atomate2.common.schemas.defects import CCDDocument
from atomate2.utils.file_client import FileClient
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.schemas.task import TaskDocument

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


class CCDInput(BaseModel):
    """Document model to help construct CCDDocument."""

    structure: Structure
    energy: float
    dir_name: str
    uuid: str


@job
def get_charged_structures(structure: Structure, charges: Iterable):
    """Add charges to a structure.

    This needs to be a job so the results of other jobs can be passed in.

    Parameters
    ----------
    structure
        A structure.
    charges
        A list of charges on the structure

    Returns
    -------
    dict
        A dictionary with the two structures with the charge states added.
    """
    structs_out = [structure.copy() for _ in charges]
    for i, q in enumerate(charges):
        structs_out[i].set_charge(q)
    return structs_out


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


@job
def get_supercell_from_prv_calc(
    uc_structure: Structure,
    prv_calc_dir: str | Path | None = None,
    sc_mat_ref: NDArray | None = None,
) -> dict:
    """Get the supercell from the previous calculation.

    Parse the previous calculation directory to obtain the supercell transformation.

    Parameters
    ----------
    uc_structure : Structure
        The unit cell structure of the bulk material.
    prv_calc_dir : Path
        The directory of the previous calculation.
    sc_mat : NDArray
        The supercell matrix. If not None, use this to validate the extracted supercell.

    Returns
    -------
    Response:
        Output containing the supercell transformation and the dir_name

    """
    fc = FileClient()
    files = fc.listdir(prv_calc_dir)
    vasprun_file = Path(prv_calc_dir) / get_zfile(files, "vasprun.xml")
    vasprun = Vasprun(vasprun_file)
    sc_structure = vasprun.initial_structure
    (sc_mat_prv, _) = get_matched_structure_mapping(
        uc_struct=uc_structure, sc_struct=sc_structure
    )

    if sc_mat_ref is not None:
        latt_ref = Lattice(sc_mat_ref)
        latt_prv = Lattice(sc_mat_prv)
        if not (
            np.allclose(sorted(latt_ref.abc), sorted(latt_prv.abc))
            and np.allclose(sorted(latt_ref.angles), sorted(latt_prv.angles))
        ):
            raise ValueError(
                "The supercell matrix extracted from the previous calculation does not match the the desired supercell shape."
            )
    return dict(sc_mat=sc_mat_prv)


@job(
    name="bulk supercell",
)
def bulk_supercell_calculation(
    uc_structure: Structure,
    relax_maker: RelaxMaker,
    sc_mat: NDArray | None = None,
) -> Response:
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
        The supercell matrix used to construct the simulation cell.

    Returns
    -------
    Response:
        Output a dictionary containing the bulk supercell calculation summary.
    """
    logger.info("Running bulk supercell calculation. Running...")
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
    return Response(output=summary_d, replace=[relax_job])


@job
def spawn_defect_calcs(
    defect: list[Defect],
    sc_mat: NDArray,
    relax_maker: RelaxMaker,
    defect_index: int | str = "",
) -> Response:
    """Spawn defect calculations from the DefectGenerator.

    Dynamic Jobflow wrapper around `run_all_charge_states`.

    Parameters
    ----------
    defect : Defect
        The defect to generate charge states for.
    sc_mat : NDArray
        The supercell matrix. If None, the code will attempt to create a
        nearly-cubic supercell.
    relax_maker : RelaxMaker
        The relax maker to be used for defect supercell calculations.
    defect_index : int | str
        Additional index to give unique names to the defect calculations.
        Useful for external bookkeeping of symmetry distinct defects.

    Returns
    -------
    Response:
        The response containing the outputs of the defect calculations as a dictionary
    """
    defect_q_jobs = []
    all_chg_outputs, add_jobs = run_all_charge_states(
        defect,
        sc_mat=sc_mat,
        relax_maker=relax_maker,
        defect_index=defect_index,
    )
    defect_q_jobs.extend(add_jobs)
    return Response(output=all_chg_outputs, replace=defect_q_jobs)


def run_all_charge_states(
    defect: Defect,
    relax_maker: RelaxMaker,
    sc_mat: NDArray | None = None,
    defect_index: int | str = "",
    add_info: dict | None = None,
) -> Response:
    """Perform charge defect supercell calculations and save the Hartree potential.

    Run a ISIF2 calculation for each available charge state of the defect.
    Ensure that the LOCPOT file is stored in the output.

    Parameters
    ----------
    defect:
        A defect object representing the defect in a unit cell.
    relax_maker:
        A RelaxMaker object to use for the atomic relaxation.
    sc_mat:
        The supercell matrix. If None, the code will attempt to create a
        nearly-cubic supercell.
    defect_index:
        Additional index to give unique names to the defect calculations.
        Useful for external bookkeeping of symmetry distinct defects.
    add_info:
        Additional information to store with the defect cell relaxation calculation.
        By default only the defect object and charge state are stored.

    Returns
    -------
    Response
        A response object containing the summary of the calculations for different
        charge states.
    """
    jobs = []
    all_chg_outputs = dict()
    sc_def_struct = defect.get_supercell_structure(sc_mat=sc_mat)
    for qq in defect.get_charge_states():
        suffix = (
            f" {defect.name} q={qq}"
            if defect_index == ""
            else f" {defect.name}-{defect_index} q={qq}"
        )
        charged_struct = sc_def_struct.copy()
        charged_struct.set_charge(qq)
        charged_relax = relax_maker.make(charged_struct)
        charged_relax.append_name(suffix)

        # write some provenances data in info.json file
        info = {
            "defect": defect,
            "charge_state": qq,
        }
        if add_info is not None:
            info.update(add_info)
        charged_relax.update_maker_kwargs(
            {"_set": {"write_additional_data->info:json": info}}, dict_mod=True
        )
        jobs.append(charged_relax)
        charged_output: TaskDocument = charged_relax.output
        all_chg_outputs[qq] = {
            "structure": charged_output.structure,
            "entry": charged_output.entry,
            "dir_name": charged_output.dir_name,
            "uuid": charged_relax.uuid,
        }
    return all_chg_outputs, jobs
