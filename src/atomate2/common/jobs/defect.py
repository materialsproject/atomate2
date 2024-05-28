"""Jobs for defect calculations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
from jobflow import Flow, Response, job
from pydantic import BaseModel
from pymatgen.analysis.defects.supercells import (
    get_matched_structure_mapping,
    get_sc_fromstruct,
)
from pymatgen.analysis.defects.thermo import DefectEntry
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

from atomate2.common.schemas.defects import CCDDocument
from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from emmet.core.tasks import TaskDoc
    from numpy.typing import NDArray
    from pymatgen.analysis.defects.core import Defect

    from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker

logger = logging.getLogger(__name__)


class CCDInput(BaseModel):
    """Document model to help construct CCDDocument."""

    structure: Structure
    energy: float
    dir_name: str
    uuid: str


@job
def get_charged_structures(structure: Structure, charges: Iterable) -> list[Structure]:
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
    for idx, q in enumerate(charges):
        structs_out[idx].set_charge(q)
    return structs_out


@job
def spawn_energy_curve_calcs(
    relaxed_structure: Structure,
    distorted_structure: Structure,
    distortions: Iterable[float],
    static_maker: StaticMaker,
    prev_dir: str | Path | None = None,
    add_name: str = "",
    add_info: dict | None = None,
) -> Response:
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
    for idx, d_struct in enumerate(distorted_structures):
        static_job = static_maker.make(d_struct, prev_dir=prev_dir)
        suffix = f" {idx}" if add_name == "" else f" {add_name} {idx}"

        # write some provenances data in info.json file
        info = {
            "relaxed_structure": relaxed_structure,
            "distorted_structure": distorted_structure,
            "distortion": s_distortions[idx],
        }
        if add_info is not None:
            info.update(add_info)
        static_job.update_maker_kwargs(
            {"_set": {"write_additional_data->info:json": info}}, dict_mod=True
        )

        static_job.append_name(f"{suffix}")
        jobs.append(static_job)
        # outputs.append(static_job.output)
        task_doc: TaskDoc = static_job.output
        outputs.append(
            {
                "structure": task_doc.output.structure,
                "energy": task_doc.output.energy,
                "dir_name": task_doc.dir_name,
                "uuid": task_doc.uuid,
            }
        )

    add_flow = Flow(jobs, outputs)
    return Response(output=outputs, replace=add_flow)


@job(output_schema=CCDDocument)
def get_ccd_documents(
    inputs1: Iterable[CCDInput],
    inputs2: Iterable[CCDInput],
    undistorted_index: int,
) -> Response:
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
    prv_calc_dir: str | Path,
    sc_entry_and_locpot_from_prv: Callable,
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
    sc_entry_and_locpot_from_prv : Callable
        Function to get the supercell ComputedStructureEntry and Locpot from the
        previous calculation.

    Returns
    -------
    Response:
        Output containing the supercell transformation and the dir_name
    """
    prv_calc_dir = strip_hostname(prv_calc_dir)
    sc_entry, plnr_locpot = sc_entry_and_locpot_from_prv(prv_calc_dir)
    sc_structure = sc_entry.structure
    sc_mat_prv, _ = get_matched_structure_mapping(
        uc_struct=uc_structure, sc_struct=sc_structure
    )

    if sc_mat_ref is not None:
        latt_ref = (uc_structure * sc_mat_ref).lattice
        latt_prv = (uc_structure * sc_mat_prv).lattice
        if not (
            np.allclose(sorted(latt_ref.abc), sorted(latt_prv.abc))
            and np.allclose(sorted(latt_ref.angles), sorted(latt_prv.angles))
        ):
            raise ValueError(
                "The supercell matrix extracted from the previous calculation "
                "does not match the the desired supercell shape."
            )
    return {
        "sc_entry": sc_entry,
        "sc_struct": sc_structure,
        "sc_mat": sc_mat_prv,
        "dir_name": prv_calc_dir,
        "lattice": Lattice(sc_structure.lattice.matrix),
        "uuid": None,
        "locpot_plnr": plnr_locpot,
    }


@job(name="bulk supercell")
def bulk_supercell_calculation(
    uc_structure: Structure,
    relax_maker: RelaxMaker,
    sc_mat: NDArray | None = None,
    get_planar_locpot: Callable | None = None,
) -> Response:
    """Bulk Supercell calculation.

    Perform a bulk supercell calculation.

    Parameters
    ----------
    uc_structure : Structure
        The unit cell structure.
    relax_maker : RelaxMaker
        The relax maker to use.
    sc_mat : NDArray | None
        The supercell matrix used to construct the simulation cell.
    get_plnr_locpot : Callable | None
        A function to get the Locpot from the output of the task document.

    Returns
    -------
    Response:
        Output a dictionary containing the bulk supercell calculation summary.
    """
    if get_planar_locpot is None:

        def get_planar_locpot(task_doc: TaskDoc) -> NDArray:
            return task_doc.calcs_reversed[0].output.locpot

    logger.info("Running bulk supercell calculation. Running...")
    sc_mat = get_sc_fromstruct(uc_structure) if sc_mat is None else sc_mat
    sc_mat = np.array(sc_mat)
    sc_structure = uc_structure * sc_mat
    relax_job = relax_maker.make(sc_structure)
    relax_job.name = "bulk relax"
    info = {"sc_mat": sc_mat.tolist()}
    relax_job.update_maker_kwargs(
        {"_set": {"write_additional_data->info:json": info}}, dict_mod=True
    )
    relax_output: TaskDoc = relax_job.output
    summary_d = {
        "uc_structure": uc_structure,
        "sc_entry": relax_output.entry,
        "sc_struct": relax_output.structure,
        "sc_mat": sc_mat.tolist(),
        "dir_name": relax_output.dir_name,
        "uuid": relax_job.uuid,
        "locpot_plnr": get_planar_locpot(relax_output),
    }
    flow = Flow([relax_job], output=summary_d)
    return Response(replace=flow)


@job
def spawn_defect_q_jobs(
    defect: Defect,
    relax_maker: RelaxMaker,
    relaxed_sc_lattice: Lattice,
    sc_mat: NDArray | None = None,
    defect_index: int | str = "",
    add_info: dict | None = None,
    validate_charge: bool = True,
    relax_radius: float | str | None = None,
    perturb: float | None = None,
) -> Response:
    """Perform charge defect supercell calculations.

    Run a atomic relaxation calculation for each available charge state of the defect.

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
    relaxed_sc_lattice:
        The lattice of the relaxed supercell. If provided, the lattice parameters
        of the supercell will be set to value specified. Otherwise, the lattice it will
        only by set by `defect.structure` and `sc_mat`.
    validate_charge:
        Whether to validate the charge states of the defect after the atomic relaxation.
        Assuming the final output of the relaxation is a TaskDoc, we should make sure
        that the charge state is set properly and matches the expected charge state from
        the input defect object.
    relax_radius:
        The radius to include around the defect site for the relaxation.
        If "auto", the radius will be set to the maximum that will fit inside a periodic
        cell. If None, all atoms will be relaxed.
    perturb:
        The amount to perturb the sites in the supercell. Only perturb the sites with
        selective dynamics set to True. So this setting only works with `relax_radius`.

    Returns
    -------
    Response
        A response object containing the summary of the calculations for different
        charge states.
    """
    defect_q_jobs = []
    all_chg_outputs = {}
    sc_def_struct = defect.get_supercell_structure(
        sc_mat=sc_mat, relax_radius=relax_radius, perturb=perturb
    )
    sc_def_struct.lattice = relaxed_sc_lattice
    if sc_mat is not None:
        sc_mat = np.array(sc_mat).tolist()
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
            "defect_name": defect.name,
            "bulk_formula": defect.structure.composition.reduced_formula,
            "bulk_num_sites": len(defect.structure),
            "bulk_space_group_info": defect.structure.get_space_group_info(),
            "sc_mat": sc_mat,
        }

        if add_info is not None:
            info.update(add_info)
        charged_relax.update_maker_kwargs(
            {"_set": {"write_additional_data->info:json": info}}, dict_mod=True
        )
        defect_q_jobs.append(charged_relax)
        charged_output: TaskDoc = charged_relax.output
        all_chg_outputs[qq] = {
            "defect": defect,
            "structure": charged_output.structure,
            "entry": charged_output.entry,
            "dir_name": charged_output.dir_name,
            "uuid": charged_relax.uuid,
            "locpot_plnr": charged_output.calcs_reversed[0].output.locpot,
        }
        # check that the charge state was set correctly
        if validate_charge:
            validation_job = check_charge_state(qq, charged_output.structure)
            defect_q_jobs.append(validation_job)
    replace_flow = Flow(defect_q_jobs, output=all_chg_outputs)
    return Response(replace=replace_flow)


@job
def check_charge_state(charge_state: int, task_structure: Structure) -> Response:
    """Check that the charge state of a defect calculation is correct.

    Parameters
    ----------
    chargestate : int
        The charge state to check.
    task_doc : TaskDoc
        The task document to check.

    Returns
    -------
    True if the charge state is correct, otherwise raises a ValueError.
    """
    if int(charge_state) != int(task_structure.charge):
        raise ValueError(
            f"The charge of the output structure is {task_structure.charge}, "
            f"but expected charge state from the Defect object is {charge_state}."
        )
    return True


@job
def get_defect_entry(charge_state_summary: dict, bulk_summary: dict) -> list[dict]:
    """Get a defect entry from a defect calculation and a bulk calculation."""
    bulk_struct_entry = bulk_summary["sc_entry"]
    # bulk_struct_entry = ComputedStructureEntry(
    #     structure=bulk_summary["sc_struct"],
    #     energy=bulk_sc_entry.energy,
    # )
    bulk_dir_name = bulk_summary["dir_name"]
    bulk_locpot = bulk_summary["locpot_plnr"]
    defect_ent_res = []
    for qq, qq_summary in charge_state_summary.items():
        defect_c_entry = qq_summary["entry"]
        defect_struct_entry = ComputedStructureEntry(
            structure=qq_summary["structure"],
            energy=defect_c_entry.energy,
        )
        defect_dir_name = qq_summary["dir_name"]
        defect_locpot = qq_summary["locpot_plnr"]
        defect_entry = DefectEntry(
            defect=qq_summary["defect"],
            charge_state=qq,
            sc_entry=defect_struct_entry,
            bulk_entry=bulk_struct_entry,
        )
        defect_ent_res.append(
            {
                "defect_entry": defect_entry,
                "defect_dir_name": defect_dir_name,
                "defect_locpot": defect_locpot,
                "bulk_dir_name": bulk_dir_name,
                "bulk_locpot": bulk_locpot,
                "bulk_uuid": bulk_summary.get("uuid"),
                "defect_uuid": qq_summary.get("uuid", None),
            }
        )
    return defect_ent_res
