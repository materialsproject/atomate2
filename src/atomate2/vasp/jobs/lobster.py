"""Module defining amset jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from jobflow import Flow,
from jobflow import Maker, Response, job
from monty.serialization import loadfn
from monty.shutil import gzip_dir
from atomate2.common.files import delete_files
from atomate2.lobster.schemas import LobsterTaskDocument
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.base import VaspInputGenerator
from atomate2.vasp.sets.core import StaticSetGenerator
from pymatgen.io.lobster import Lobsterin
from atomate2.vasp.powerups import update_user_incar_settings
from pymatgen.core import Structure
from atomate2.lobster.jobs import PureLobsterMaker
from atomate2.utils.path import strip_hostname

__all__ = ["VaspLobsterMaker", "get_basis_infos", "get_lobster_jobs"]

logger = logging.getLogger(__name__)

# include a class where we can also add Lobster


@dataclass
class VaspLobsterMaker(BaseVaspMaker):
    """
    Maker that performs a VASP computation with settings that are required for Lobter runs
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
    # TODO: set grid_density to a normal value
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"grid_density": 1},
            user_incar_settings={
                "IBRION": 2,
                "ISIF": 2,
                "ENCUT": 700,
                "EDIFF": 1e-7,
                "LAECHG": False,
                "LREAL": False,
                "ALGO": "Normal",
                "NSW": 99,
                "LCHARG": False,
                "LWAVE": True,
                "ISYM": 0,
            },
        )
    )
    copy_vasp_kwargs = {"additional_vasp_files": ["WAVECAR"]}


@job
def get_basis_infos(
    structure, vaspmaker, address_max_basis=None, address_min_basis=None
):
    # create a vasp input for lobster

    # we need to add more files to copy if prev_vasp_dir exists
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
def update_user_incar_settings_job(vaspjob, nbands):
    vaspjob = update_user_incar_settings(vaspjob, {"NBANDS": nbands["nbands"]})
    return Response(replace=vaspjob)


@job
def get_lobster_jobs(basis_dict, wavefunction_dir):
    jobs = []
    outputs = {}
    outputs["uuids"] = []
    outputs["dirs"] = []
    outputs["basis"] = []
    for i, basis in enumerate(basis_dict):
        lobsterjob = PureLobsterMaker(name="lobster_run_{}".format(i)).make(
            wavefunction_dir=wavefunction_dir, basis_dict=basis
        )
        outputs["uuids"].append(lobsterjob.output.uuid)
        outputs["dirs"].append(lobsterjob.output.dir_name)
        outputs["basis"].append(basis)
        jobs.append(lobsterjob)

    flow = Flow(jobs, output=outputs)
    return Response(replace=flow)


@job
def delete_lobster_wavecar(dirs, dir_vasp=None, dir_preconverge=None):
    jobs = []
    outputs = {}
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
