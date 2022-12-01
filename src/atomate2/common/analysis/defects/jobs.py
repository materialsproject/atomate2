"""Jobs for defect calculations."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from jobflow import Response, job
from numpy.typing import NDArray
from pymatgen.analysis.defects.core import Defect
from pymatgen.analysis.defects.supercells import (
    get_matched_structure_mapping,
    get_sc_fromstruct,
)
from pymatgen.analysis.defects.thermo import DefectEntry
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Vasprun

from atomate2.common.analysis.defects.schemas import FormationEnergyDiagramDocument
from atomate2.common.files import get_zfile
from atomate2.utils.file_client import FileClient
from atomate2.utils.path import strip_hostname
from atomate2.vasp.jobs.core import RelaxMaker
from atomate2.vasp.schemas.task import TaskDocument

logger = logging.getLogger(__name__)


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


@job(output_schema=FormationEnergyDiagramDocument)
def collect_defect_outputs(
    defect,
    all_chg_outputs: dict,
    bulk_sc_dir: str,
    dielectric: float | NDArray | None = None,
) -> dict:
    """Collect all the outputs from the defect calculations.

    This job will combine the structure and entry fields to create a
    ComputerStructureEntry object.

    Parameters
    ----------
    defect:
        The defect object.
    all_chg_outputs:
        The output from the defect calculations. See `run_all_charge_states` for
        more information.
    bulk_sc_dir:
        The directory containing the bulk supercell calculation.
    dielectric:
        The dielectric constant used to construct the formation energy diagram.
        If None (default), finite size corrections will not be applied.
    """

    def get_locpot_from_dir(dir_name: str) -> Locpot:
        locpot_path = Path(strip_hostname(dir_name)) / "LOCPOT.gz"
        return Locpot.from_file(locpot_path)

    def parse_bulk_dir(dir_name: str) -> dict:
        vbm_path = Path(strip_hostname(dir_name)) / "vasprun.xml.gz"
        vasp_run = Vasprun(vbm_path)
        band_structure = vasp_run.get_band_structure()
        entry = vasp_run.get_computed_entry()
        return dict(entry=entry, band_structure=band_structure)

    bulk_locpot = get_locpot_from_dir(bulk_sc_dir)
    bulk_data = parse_bulk_dir(bulk_sc_dir)
    logger.info(f"Bulk entry energy: {bulk_data['entry'].energy} eV")

    fe_doc = FormationEnergyDiagramDocument(
        bulk_entry=bulk_data["entry"],
        vbm=bulk_data["band_structure"].get_vbm()["energy"],
        band_gap=bulk_data["band_structure"].get_band_gap()["energy"],
        bulk_sc_dir=bulk_sc_dir,
        dielectric=dielectric,
        defect_entries=list(),
        defect_sc_dirs=dict(),
    )

    # loop over the different distinct defect: Mg_Ga_1, Mg_Ga_2, ...
    logger.debug(f"Processing defect {defect.name}")
    defect_locpots = dict()
    # defect_entries: list[DefectEntry] = []
    # then loop over the different charge states
    for qq, v in all_chg_outputs.items():
        logger.debug(f"Processing charge state {qq}")
        if not isinstance(v, dict):
            continue
        defect_locpots[int(qq)] = get_locpot_from_dir(v["dir_name"])
        sc_dict = v["entry"].as_dict()
        sc_dict["structure"] = v["structure"]
        sc_entry = ComputedStructureEntry.from_dict(sc_dict)
        def_ent = DefectEntry(
            defect=defect,
            charge_state=int(qq),
            sc_entry=sc_entry,
        )
        def_ent.get_freysoldt_correction(
            defect_locpots[int(qq)],
            bulk_locpot,
            dielectric=dielectric,
        )
        fe_doc.defect_entries.append(def_ent)
        fe_doc.defect_sc_dirs[qq] = v["dir_name"]
    return fe_doc
