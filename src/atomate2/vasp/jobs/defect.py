"""Jobs for defect calculations."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

import numpy as np
from jobflow import Flow, Response, job
from numpy.typing import NDArray
from pydantic import BaseModel
from pymatgen.analysis.defect.core import Defect
from pymatgen.analysis.defect.generators import DefectGenerator
from pymatgen.analysis.defect.supercells import (
    get_matched_structure_mapping,
    get_sc_fromstruct,
)
from pymatgen.analysis.defect.thermo import DefectEntry
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp import Incar
from pymatgen.io.vasp.inputs import Kpoints, Kpoints_supported_modes
from pymatgen.io.vasp.outputs import WSWQ, Locpot, Vasprun

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
DEFECT_KPOINT_SETTINGS = Kpoints(
    comment="special k-point",
    num_kpts=1,
    style=Kpoints_supported_modes.Reciprocal,
    kpts=((0.25, 0.25, 0.25),),
    kpts_shift=(0, 0, 0),
    kpts_weights=[1],
)

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


@job(
    name="bulk supercell",
)
def bulk_supercell_calculation(
    uc_structure: Structure,
    relax_maker: RelaxMaker,
    sc_mat: NDArray | None = None,
):
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

    Returns
    -------
    dict:
        The bulk supercell calculation summary.
    """
    _logger.info("Running bulk supercell calculation. Running...")
    sc_mat = get_sc_fromstruct(uc_structure) if sc_mat is None else sc_mat
    sc_mat = np.array(sc_mat)
    sc_structure = uc_structure * sc_mat
    relax_job = relax_maker.make(sc_structure)
    relax_job.name = "bulk relax"
    relax_output: TaskDocument = relax_job.output

    # Is this the best way to pass a subset of data through?
    summary_d = dict(
        uc_structure=uc_structure,
        sc_entry=relax_output.entry,
        sc_struct=relax_output.structure,
        sc_mat=sc_mat.tolist(),
        dir_name=relax_output.dir_name,
        uuid=relax_job.uuid,
    )
    waiter = wait_for_dict(summary_d)
    return Response(output=waiter.output, replace=[relax_job, waiter])


@job
def wait_for_dict(d: dict):
    """Wait for a dictionary to be populated.

    Parameters
    ----------
    d : dict
        The dictionary to wait for.
    """
    return d


# @job
# def create_summary(d: dict) -> BulkSuperCellSummary:
#     """Create a summary from a bulk supercell calculation."""
#     entry = d["sc_entry"]
#     structure = d["sc_struct"]
#     entry_d = entry.as_dict()
#     entry_d["structure"] = structure.as_dict()
#     d["sc_entry"] = ComputedStructureEntry.from_dict(entry_d)
#     return BulkSuperCellSummary(**d)


@job
def spawn_defects_calcs(
    defect_gen: DefectGenerator,
    sc_mat: NDArray,
    relax_maker: RelaxMaker,
    bulk_sc_dir: str | Path | None,
) -> Response:
    """Spawn defect calculations from the DefectGenerator.

    Parameters
    ----------
    defect_gen : DefectGenerator
        The defect generator to use.
    sc_mat : NDArray
        The supercell matrix. If None, the code will attempt to create a nearly-cubic supercell.
    bulk_sc_dir : str | Path | None
        The directory of the bulk supercell calculation.
        If the directory name is "skip", we will not store the pristine supercell.

    Returns
    -------
    Response:
        The response containing the outputs of the defect calculations as a dictionary
    """
    if bulk_sc_dir is not None:
        bulk_sc_dir = strip_hostname(bulk_sc_dir)
        bulk_sc_dir = Path(bulk_sc_dir)
        bulk_sc_entry = Vasprun(bulk_sc_dir / "vasprun.xml.gz").get_computed_entry(
            inc_structure=True
        )
        # bulk_locpot = Locpot(bulk_sc_dir / "LOCPOT.gz")
        (sc_mat, _) = get_matched_structure_mapping(
            defect_gen.structure, bulk_sc_entry.structure
        )  # TODO: this might need relaxing if too tight

    defect_q_jobs = []
    output = dict()
    name_counter: dict = defaultdict(lambda: 0)

    for defect in defect_gen:
        defect_job = perform_defect_calcs(
            defect,
            sc_mat=sc_mat,
            relax_maker=relax_maker,
            prev_vasp_dir=str(bulk_sc_dir),
            defect_index=f"{name_counter[defect.name]}",
        )
        defect_q_jobs.append(defect_job)
        output[f"{defect.name}-{name_counter[defect.name]}"] = dict(
            defect=defect,
            results=defect_job.output,
        )
        name_counter[defect.name] += 1
    return Response(output=output, replace=defect_q_jobs)


@job
def perform_defect_calcs(
    defect: Defect,
    relax_maker: RelaxMaker | None = None,
    prev_vasp_dir: str | Path | None = None,
    sc_mat: NDArray | None = None,
    defect_index: str = "",
    add_info: dict | None = None,
) -> Response:
    """Perform charge defect supercell calculations and save the Hartree potential.

    Parameters
    ----------
    defect:
        A defect object representing the defect in a unit cell.
    relax_maker:
        A RelaxMaker object to use for the atomic relaxation. If None, the default will be used (see DEFAULT_RELAX_MAKER).
    prev_vasp_dir:
        The directory containing the previous VASP calculation.
    sc_mat:
        The supercell matrix. If None, the code will attempt to create a nearly-cubic supercell.
    defect_index:
        Additional index to give unique names to the defect calculations.
    add_info:
        Additional information to store with the defect cell relaxation calculation.
        By default only the defect object and charge state are stored.

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
        charge_output: TaskDocument = charged_relax.output

        outputs[qq] = {
            "structure": charge_output.structure,
            "entry": charge_output.entry,
            "dir_name": charge_output.dir_name,
            "uuid": charged_relax.uuid,
        }
    outputs["defect"] = defect
    add_flow = Flow(jobs, output=outputs)
    return Response(output=outputs, replace=add_flow)


@job
def collect_defect_outputs(
    defects_output: dict, bulk_sc_dir: str, dielectric: float | NDArray
) -> dict:
    """Collect all the outputs from the defect calculations.

    This job will combine the structure and entry fields to create a ComputerStructureEntry object.
    """

    def get_locpot_from_dir(dir_name: str) -> Locpot:
        locpot_path = Path(strip_hostname(dir_name)) / "LOCPOT.gz"
        return Locpot.from_file(locpot_path)

    def parse_bulk_dir(dir_name: str) -> dict:
        vbm_path = Path(strip_hostname(dir_name)) / "vasprun.xml.gz"
        vasp_run = Vasprun(vbm_path)
        vbm = vasp_run.get_band_structure().get_vbm()
        entry = vbm.get_computed_entry()
        return dict(entry=entry, vbm=vbm)

    bulk_data = parse_bulk_dir(bulk_sc_dir)
    output = dict(
        defects_output=defects_output,
        bulk_entry=bulk_data["entry"],
        bulk_vbm=bulk_data["vbm"],
    )
    # first loop over the different distinct defect: Mg_Ga_1, Mg_Ga_2, ...
    for defect_name, def_out in defects_output.items():
        defect = def_out.pop("defect")
        defect_locpots = dict()
        bulk_locpot = get_locpot_from_dir(bulk_sc_dir)
        defect_entries = []
        # then loop over the different charge states
        for qq, v in def_out.items():
            defect_locpots[qq] = get_locpot_from_dir(v["dir_name"])
            sc_dict = v["entry"].as_dict()
            sc_dict["structure"] = v["structure"]
            sc_entry = ComputedStructureEntry.from_dict(sc_dict)
            def_ent = DefectEntry(
                defect=defect, charge_state=qq, sc_entry=sc_entry, dielectric=dielectric
            )
            def_ent.get_freysoldt_correction(defect_locpots[qq], bulk_locpot)
            defect_entries.append(def_ent)
        output[defect_name] = dict(
            defect=defect,
            defect_entries=defect_entries,
        )
    return output


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
    distorted_calc_dirs: List[str],
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
