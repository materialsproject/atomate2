"""Module defining amset jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Response, job
from pymatgen.core import Structure
from pymatgen.io.lobster import Lobsterin

from atomate2.common.files import delete_files
from atomate2.lobster.jobs import PureLobsterMaker
from atomate2.lobster.schemas import LobsterTaskDocument
from atomate2.utils.path import strip_hostname
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.powerups import update_user_incar_settings
from atomate2.vasp.sets.base import VaspInputGenerator
from atomate2.vasp.sets.core import StaticSetGenerator

__all__ = ["VaspLobsterMaker", "get_basis_infos", "get_lobster_jobs"]

logger = logging.getLogger(__name__)


# include a class where we can also add Lobster


@dataclass
class VaspLobsterMaker(BaseVaspMaker):
    """
    Maker that performs a VASP computation with
     settings that are required for Lobter runs.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "static_run"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"grid_density": 6000},
            user_incar_settings={
                "IBRION": 2,
                "ISIF": 2,
                "ENCUT": 680,
                "EDIFF": 1e-7,
                "LAECHG": False,
                "LREAL": False,
                "ALGO": "Normal",
                "NSW": 0,
                "LCHARG": False,
                "LWAVE": True,
                "ISYM": 0,
            },
        )
    )
    copy_vasp_kwargs: dict = field(
        default_factory=lambda: {"additional_vasp_files": ["WAVECAR"]}
    )


@job
def get_basis_infos(
    structure: Structure,
    vaspmaker: BaseVaspMaker,
    address_max_basis: str = None,
    address_min_basis: str = None,
):
    """
    Compute all relevant basis sets and maximum number of bands.

    Parameters
    ----------
    structure: Structure object.
    vaspmaker: BaseVaspMaker.
    address_max_basis: string to yaml file including basis set information.
    address_min_basis: string to yaml file including basis set information.

    Returns
    -------
        dict including number of bands and basis set information
    """
    potcar_symbols = vaspmaker.input_set_generator._get_potcar(
        structure=structure, potcar_spec=True
    )

    # get data from Lobsterinput
    # Lobster
    if address_max_basis is None and address_min_basis is None:
        list_basis_dict = Lobsterin.get_all_possible_basis_functions(
            structure=structure, potcar_symbols=potcar_symbols
        )
    elif address_max_basis is not None and address_min_basis is None:
        list_basis_dict = Lobsterin.get_all_possible_basis_functions(
            structure=structure,
            potcar_symbols=potcar_symbols,
            address_basis_file_max=address_max_basis,
        )
    elif address_min_basis is not None and address_max_basis is None:
        list_basis_dict = Lobsterin.get_all_possible_basis_functions(
            structure=structure,
            potcar_symbols=potcar_symbols,
            address_basis_file_min=address_min_basis,
        )
    elif address_min_basis is not None and address_max_basis is not None:
        list_basis_dict = Lobsterin.get_all_possible_basis_functions(
            structure=structure,
            potcar_symbols=potcar_symbols,
            address_basis_file_max=address_max_basis,
            address_basis_file_min=address_min_basis,
        )

    nband_list = []
    for dict_for_basis in list_basis_dict:
        basis = [key + " " + value for key, value in dict_for_basis.items()]
        lobsterin = Lobsterin(settingsdict={"basisfunctions": basis})
        nbands = lobsterin._get_nbands(structure=structure)
        nband_list.append(nbands)

    return {"nbands": max(nband_list), "basis_dict": list_basis_dict}


@job
def update_user_incar_settings_maker(
    vaspmaker: VaspLobsterMaker,
    nbands: dict,
    structure: Structure,
    prev_vasp_dir: Path | str,
):
    """
    Update the INCAR settings of a maker.

    Parameters
    ----------
    vaspmaker: VaspLobsterMaker.
    nbands: int indicating the correct number of bands
    structure: Structure object.
    prev_vasp_dir: Path or string to vasp files.

    Returns
    -------
        VaspLobsterMaker with correct number of bands.
    """
    vaspmaker = update_user_incar_settings(vaspmaker, {"NBANDS": nbands})
    vaspjob = vaspmaker.make(structure=structure, prev_vasp_dir=prev_vasp_dir)

    return Response(replace=vaspjob)


@job
def get_lobster_jobs(
    lobstermaker,
    basis_dict,
    wavefunction_dir,
    optimization_run_job_dir,
    optimization_run_uuid,
    static_run_job_dir,
    static_run_uuid,
    additional_static_run_job_dir,
    additional_static_run_uuid,
):
    """
    Create a list of Lobster jobs with different basis sets.

    Parameters
    ----------
    lobstermaker: maker for the Lobster jobs
    basis_dict: dict including basis set information.
    wavefunction_dir: Path to VASP calculation with WAVECAR
    optimization_run_job_dir: Path to optimization run.
    optimization_run_uuid: uuid of optimization run.
    static_run_job_dir: Path to static VASP calculation.
    static_run_uuid: uuid of static run.
    additional_static_run_job_dir: Path to preconvergence step.
    additional_static_run_uuid: uuid of preconvergence step.

    Returns
    -------
        List of Lobster jobs.
    """
    jobs = []
    outputs = {}
    outputs["optimization_run_job_dir"] = optimization_run_job_dir
    outputs["optimization_run_uuid"] = optimization_run_uuid
    outputs["static_run_job_dir"] = static_run_job_dir
    outputs["static_run_uuid"] = static_run_uuid
    outputs["additional_static_run_dir"] = additional_static_run_job_dir
    outputs["additional_static_uuid"] = additional_static_run_uuid
    outputs["lobster_uuids"] = []
    outputs["lobster_dirs"] = []
    outputs["lobster_task_documents"] = []

    if lobstermaker is None:
        lobstermaker = PureLobsterMaker()

    for i, basis in enumerate(basis_dict):
        lobsterjob = lobstermaker.make(
            wavefunction_dir=wavefunction_dir, basis_dict=basis
        )
        lobsterjob.append_name(f"_run_{i}")
        outputs["lobster_uuids"].append(lobsterjob.output.uuid)
        outputs["lobster_dirs"].append(lobsterjob.output.dir_name)
        outputs["lobster_task_documents"].append(lobsterjob.output)
        jobs.append(lobsterjob)

    flow = Flow(jobs, output=outputs)
    return Response(replace=flow)


@job
def delete_lobster_wavecar(
    dirs: list, dir_vasp: Path | str = None, dir_preconverge: Path | str = None
):
    """
    Delete all WAVECARs.

    Parameters
    ----------
    dirs: Path to directories of lobster jobs.
    dir_vasp: Path to directory of static VASP run.
    dir_preconverge: Path to directory of preconvergence run.
    """
    jobs = []
    outputs: dict = {}
    outputs["lobster_dir_name"] = []
    outputs["add_static_dir_name"] = []
    outputs["static_dir_name"] = []
    dec_delete_files = job(delete_files)
    for dir_name in dirs:
        jobs.append(
            dec_delete_files(
                dir_name, include_files=["WAVECAR", "WAVECAR.gz"], allow_missing=True
            )
        )
        outputs["lobster_dir_name"].append(dir_name)
    if dir_preconverge is None and dir_vasp is not None:
        dir_vasp_stat = strip_hostname(dir_vasp)
        jobs.append(
            dec_delete_files(
                dir_vasp_stat,
                include_files=["WAVECAR", "WAVECAR.gz"],
                allow_missing=True,
            )
        )
        outputs["static_dir_name"].append(dir_vasp_stat)
    if dir_preconverge is not None and dir_vasp is not None:
        dir_vasp_add_stat = strip_hostname(dir_preconverge)
        dir_vasp_stat = strip_hostname(dir_vasp)
        jobs.append(
            dec_delete_files(
                dir_vasp_stat,
                include_files=["WAVECAR", "WAVECAR.gz"],
                allow_missing=True,
            )
        )
        jobs.append(
            dec_delete_files(
                dir_vasp_add_stat,
                include_files=["WAVECAR", "WAVECAR.gz"],
                allow_missing=True,
            )
        )
        outputs["add_static_dir_name"].append(dir_vasp_add_stat)
        outputs["static_dir_name"].append(dir_vasp_stat)

    flow = Flow(jobs, output=outputs)
    return Response(replace=flow)


@job(output_schema=LobsterTaskDocument)
def generate_database_entry(
    **kwargs,
):
    """
    Analyze the LOBSTER runs and summarize the results.

    Parameters
    ----------
    kwargs: dict
        Additional parameters that are passed to LobsterTaskDocument.from_directory

    """
    lobster_doc = LobsterTaskDocument.from_directory(
        **kwargs,
    )

    return lobster_doc
