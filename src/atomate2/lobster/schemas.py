"""Module defining amset document schemas."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from pymatgen.core import Structure
from pymatgen.io.lobster import Lobsterin, Lobsterout, Icohplist

import os
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from pydantic import BaseModel, Field


from atomate2 import __version__
from atomate2.common.schemas.math import Matrix3D, Vector3D
from atomate2.common.schemas.structure import StructureMetadata
from atomate2.utils.datetime import datetime_str
from atomate2.utils.path import get_uri

from atomate2.common.files import get_zfile


try:
    from lobsterpy.cohp.analyze import Analysis
    from lobsterpy.cohp.describe import Description
except ImportError:
    Analysis=None
    Description=None

__all__ = ["LobsterTaskDocument"]

logger = logging.getLogger(__name__)

class LobsteroutModel(BaseModel):
    """Collection to store computational settings from the LOBSTER computation."""

    restart_from_projection: bool =Field("has this run been restarted from a projection")
    lobster_version: str = Field("Lobster version")
    threads: int = Field("Number of threads that Lobster ran on")
    dft_program: str = Field("Which DFT program was used for this run")
    chargespilling: float = Field("Absolute charge spilling")
    totalspilling: float = Field("Total spilling")
    elements: str = Field("Elements in structure")
    basistype: str= Field("Basis set used in Lobster")
    basisfunctions: str=Field("basis_functions")
    timing: str=Field("timing")
    warnings: str = Field("Warnings")
    orthonormalization: str = Field("info_orthonormalization")
    infos: str = Field("info_lines")
    hasDOSCAR: bool =Field("Bool indicating if DOSCAR is present.")
    hasCOHPCAR: bool =Field("Bool indicating if COHPCAR is present.")
    hasCOOPCAR: bool = Field("Bool indicating if COOPCAR is present.")
    hasCOBICAR: bool = Field("Bool indicating if COBICAR is present.")
    hasCHARGE: bool = Field("Bool indicating if CHARGE is present.")
    hasmadelung: bool  = Field("Bool indicating if Madelung file is present.")
    hasProjection: bool = Field("Bool indicating if projection file is present.")
    hasbandoverlaps: bool = Field("Bool indicating if BANDOVERLAPS file is presetn")
    hasfatband: bool = Field("Bool indicating if Fatbands are present.")
    hasGrossPopulation: bool = Field("Bool indicating if GrossPopulations file is present.")
    hasDensityOfEnergies: bool = Field("Bool indicating if DensityofEnergies is present")

class LobsterinModel(BaseModel):
    """Collection to store input settings for the LOBSTER computation."""

    COHPstartEnergy: bool =Field("has this run been restarted from a projection")
    COHPendEnergy: str = Field("Lobster version")
    COHPSteps: int = Field("Number of threads that Lobster ran on")
    basisSet: str = Field("Which DFT program was used for this run")
    cohpGenerator: float = Field("Absolute charge spilling")
    saveProjectionToFile: float = Field("Total spilling")
    basisfunctions: str = Field("Elements in structure")

class CondensedBondingAnalysisModel(BaseModel):
    """Collection to store condensed bonding analysis data from LobsterPy based on ICOHP"""

    formula: str = Field("Pretty formula of the structure")
    max_considered_bond_length: float = Field("Maximum bond length considered in bonding analysis")
    limit_icohp: List[float] =Field("ICOHP range considered in co-ordination environment analysis")
    number_of_considered_ions: int = Field("number of ions detected based on Mulliken/Löwdin Charges")
    sites: dict = Field("Dict object that describes bond summary stats, "
                        "bonding/antibonding percentage and its coordination environment")
    type_charges: str =Field("Charge type considered for assinging valences in bonding analysis")
    madelung_energy: float = Field("Total electrostatic energy for the structure based on chosen type_charges")
    cutoff_icohp: float = Field("Percent limiting the ICOHP values to be considered relative to strongest ICOHP")
    summed_spins: bool = Field("Bool stating whether to sum spin channels during analysis")
    start: Optional[float] = Field("Sets the lower limit of energy relative to Fermi for evaluating"""
                         "bonding/anti-bonding percentages in the bond"
                         "if set to None, all energies up-to the Fermi is considered")
    cohp_plot_data:dict[float] = Field("Stores the COHP plot data based on relevant bond labels "
                                       "for site as keys")

class StrongestBondsModel(BaseModel):
    """Collection to store strongest bonds extracted from ICOHPLIST/ICOOPLIST/ICOBILIST data from LOBSTER runs"""

    only_cation_anion: bool = Field("If True, only information of cation-anion pairs "
                                    "bond strength ,length will be returned ")
    are_coops: bool = Field("Denotes whether the file consists of ICOOPs")
    are_cobis: bool = Field("Denotes whether the file consists of ICOBIs")
    bond_label: str = Field("String denoting atom pairs")
    bond_strength: float = Field("Strongest bond based on ICOHPLIST/ICOOPLIST/ICOBILIST")
    bond_length: float = Field("Bond length corresponding strongest bond based on ICOHPLIST/ICOOPLIST/ICOBILIST")



class LobsterTaskDocument(StructureMetadata):
    """Definition of LOBSTER task document."""

    dir_name: str = Field(None, description="The directory for this Lobster task")
    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp for this task document was last updated",
    )
    completed_at: str = Field(
        None, description="Timestamp for when this task was completed"
    )
    lobsterout: LobsteroutModel =Field("Lobster out data")
    lobsterin: Lobsterin = Field("Lobsterin")
    #LobsterPy_cation_anion: LobsterPyModel = Field("Model describing the LobsterPy data")
    #COHPData
    #COOPData
    #COBIData


    structure: Structure = Field(None, description="The structure used in this task")
    _schema: str = Field(
        __version__,
        description="Version of atomate2 used to create the document",
        alias="schema",
    )

    @classmethod
    @requires(Analysis, "lobsterpy must be installed to create an LobsterTaskDocument.")
    def from_directory(
        cls,
        dir_name: Union[Path, str],
        additional_fields: Dict[str, Any] = None,
    ):
        """
        Create a task document from a directory containing LOBSTER files.

        Parameters
        ----------
        dir_name : path or str
            The path to the folder containing the calculation outputs.
        additional_fields : dict
            Dictionary of additional fields to add to output document.

        Returns
        -------
        AmsetTaskDocument
            A task document for the lobster calculation.
        """

        additional_fields = {} if additional_fields is None else additional_fields
        dir_name = Path(dir_name)
        # do automatic analysis with lobsterpy and provide data

        #struct = Structure.from_file(get_zfile(directory_listing=".",base_name="POSCAR"))
        Lobsterout_here = Lobsterout("lobsterout.gz")
        lobsterout_doc=Lobsterout_here.get_doc()
        #Lobsterin_here = Lobsterin.from_file(get_zfile(directory_listing=".", base_name="lobsterin"))
        # cation anion-mode

        # analyse = Analysis(
        #     path_to_poscar=os.path.join(directory, "POSCAR"),
        #     path_to_icohplist=os.path.join(directory, "ICOHPLIST.lobster"),
        #     path_to_cohpcar=os.path.join(directory, "COHPCAR.lobster"),
        #     path_to_charge=os.path.join(directory, "CHARGE.lobster"),
        #     summed_spins=True,
        #     cutoff_icohp=0.01
        # )
        #
        # # Setup Desciption dict
        # describe = Description(analysis_object=analyse)
        # describe.write_description()
        #
        # # Automatic plots
        # # describe.plot_cohps(
        # #    ylim=[-4, 2],
        # #    xlim=[-10, 10],
        # #    integrated=False,
        # # )
        #
        # # different dicts that summarize the results
        #
        # print(analyse.condensed_bonding_analysis)
        # print(analyse.final_dict_bonds)
        # print(analyse.final_dict_ions)
        #
        # # This is similar to the ChemEnv output now
        # lse = analyse.chemenv.get_light_structure_environment(only_cation_environments=True)
        #
        # for coord, neighbor_sets in zip(lse.coordination_environments, lse.neighbors_sets):
        #     # we are only looking at cation-anion bonds here and we only return environments of cations
        #     if neighbor_sets is not None:
        #         print(coord[0])
        #         print(neighbor_sets[0])
        #         # This returns the list of all neighboring atoms as PeriodicSite objects including coordinates.
        #         # This should be useful for computing vectors
        #         print(neighbor_sets[0].neighb_sites)
        #
        # # process Lobsterout (similar to outputs from atomate)
        #
        # # use lobsterpy for automatic analysis
        # # add further infos on COHP

        #doc.copy(update=additional_fields)
        return  lobsterout_doc #doc


def _get_structure() -> Structure:
    """Find amset input file in current directory and extract structure."""
    vr_files = list(Path(".").glob("*vasprun.xml*"))
    bs_files = list(Path(".").glob("*band_structure_data*"))

    if len(vr_files) > 0:
        from pymatgen.io.vasp import BSVasprun

        return BSVasprun(str(vr_files[0])).get_band_structure().structure
    elif len(bs_files) > 0:
        return loadfn(bs_files[0])["band_structure"].structure

    raise ValueError("Could not find amset input in current directory.")
