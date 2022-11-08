"""Module defining amset jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from jobflow import Flow, Maker
from jobflow import Maker, Response, job
from monty.serialization import loadfn
from monty.shutil import gzip_dir
from atomate2.lobster.schemas import LobsterTaskDocument
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.base import VaspInputGenerator
from atomate2.vasp.sets.core import StaticSetGenerator
from pymatgen.io.lobster import Lobsterin
from atomate2.vasp.powerups import update_user_incar_settings
from pymatgen.core import Structure
from atomate2.lobster.jobs import PureLobsterMaker
__all__ = [ "VaspLobsterMaker"]

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

    name: str = "lobster"
    #TODO: set grid_density to a normal value
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
                "LWAVE":True,
                "ISYM":0
            },
        )
    )

@job
def get_vasp_lobster_jobs(structure, address_max_basis, address_min_basis, prev_vasp_dir):
    # create a vasp input for lobster
    vaspmaker=VaspLobsterMaker()
    # we need to add more files to copy if prev_vasp_dir exists
    potcar_symbols=vaspmaker.input_set_generator._get_potcar(structure=structure, potcar_spec=True)
    vaspjob=vaspmaker.make(structure=structure)
    # get data from Lobsterinput
    #Lobster
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
    for basis in list_basis_dict:
        lobsterin = Lobsterin(settingsdict={"basisfunctions": basis})
        nbands = lobsterin._get_nbands(structure=structure)
        nband_list.append(nbands)

    #vaspjob.maker.input_set_generator.user_incar_settings["NBANDS"] = max(nband_list)

    jobs=[]
    jobs.append(vaspjob)
    update_user_incar_settings(vaspjob, {"NBANDS": max(nband_list)})

    for basis in list_basis_dict:
        jobs.append(PureLobsterMaker().make(wavefunction_dir=vaspjob.output.dir_name, basis_dict=basis))

    #create a list of lobsterjobs or give back the different basis sets that need to be computed?!
    # make lobster jobs
    flow=Flow(jobs)

    #make a flow of vaspjob and LobsterJobs
    # return a vasp job and set of lobster computations
    return Response(replace=flow)

#
#
# @dataclass
# class PureLobsterMaker(Maker):
#     """
#     Lobster job maker.
#
#     Parameters
#     ----------
#     name : str
#         Name of jobs produced by this maker.
#     resubmit : bool
#         Could this be interesting?
#     task_document_kwargs : dict
#         Keyword arguments passed to :obj:`.LobsterTaskDocument.from_directory`.
#     """
#
#     name: str = "lobster"
#     resubmit: bool = False
#     task_document_kwargs: dict = field(default_factory=dict)
#
#     @job(output_schema=LobsterTaskDocument)
#     def make(
#         self,
#         settings: dict,
#         #prev_lobster_dir: str | Path = None, # needed?
#         wavefunction_dir: str | Path = None,
#     ):
#         """
#         Run an AMSET calculation.
#
#         Parameters
#         ----------
#         settings : dict
#             Amset settings.
#         wavefunction_dir : str or Path
#             A directory containing a VASP computation including WAVECAR
#             # could be extended to other codes as well
#
#         """
#         copy_lobster_files(wavefunction_dir)
#
#         # write amset settings
#         write_lobster_settings(settings)
#
#         # run amset
#         logger.info("Running LOBSTER")
#         run_lobster()
#
#
#         # what checks might be useful? we have validators in custodian already
#
#
#
#         # parse amset outputs
#         task_doc = LosterTaskDocument.from_directory(
#             Path.cwd(), **self.task_document_kwargs
#         )
#
#         # gzip folder
#         gzip_dir(".")
#
#         # handle resubmission for non-converged calculations
#         # not sure what do here or if needed!
#
#
#         return Response(output=task_doc)

