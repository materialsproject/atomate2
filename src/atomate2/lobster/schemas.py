"""Module defining amset document schemas."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.lobster import Lobsterin, Lobsterout

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
    lobsterpy = None

__all__ = ["LobsterTaskDocument"]

logger = logging.getLogger(__name__)

class LobsteroutModel(BaseModel):
    """Collection to store computational settings for the phonon computation."""

    restart_from_projection: bool =Field("has this run been restarted from a projection")
    lobster_version: str = Field("Lobster version")
    threads: int = Field("Number of threads that Lobster ran on")
    dft_program: str = Field("Which DFT program was used for this run")
    chargespilling: float = Field("Charge spilling")
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

class CondensedBondingAnalysisModel(BaseModel):
    """Collection to store computational settings for the phonon computation."""
    #formula: formula,
    #"max_considered_bond_length": max_bond_lengths,
    #"limit_icohp": limit_icohps,
    #"number_of_considered_ions": number_considered_ions,
    #"sites": site_dict,
    #"type_charges": self.type_charge,
    #"madelung_energy": madelung_energy,


# get stuff from VASP?



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
    LobsterPy_cation_anion: LobsterPyModel = Field("Model describing the LobsterPy data")
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
    @requires(lobsterpy, "lobsterpy must be installed to create an AmsetTaskDocument.")
    def from_directory(
        cls,
        dir_name: Union[Path, str],
        additional_fields: Dict[str, Any] = None,
    ):
        """
        Create a task document from a directory containing VASP files.

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

        struct = Structure.from_file(get_zfile("POSCAR"))
        Lobsterout_here = Lobsterout(get_zfile("lobsterout"))
        lobsterout_doc=Lobsterout_here.get_doc()
        Lobsterin_here = Lobsterin.from_file(get_zfile("lobsterin"))
        # cation anion-mode

        directory = (
                Path(__file__).parent / "LobsterOutputs/outputs"
        )

        analyse = Analysis(
            path_to_poscar=os.path.join(directory, "POSCAR"),
            path_to_icohplist=os.path.join(directory, "ICOHPLIST.lobster"),
            path_to_cohpcar=os.path.join(directory, "COHPCAR.lobster"),
            path_to_charge=os.path.join(directory, "CHARGE.lobster"),
            summed_spins=True,
            cutoff_icohp=0.01
        )

        # Setup Desciption dict
        describe = Description(analysis_object=analyse)
        describe.write_description()

        # Automatic plots
        # describe.plot_cohps(
        #    ylim=[-4, 2],
        #    xlim=[-10, 10],
        #    integrated=False,
        # )

        # different dicts that summarize the results

        print(analyse.condensed_bonding_analysis)
        print(analyse.final_dict_bonds)
        print(analyse.final_dict_ions)

        # This is similar to the ChemEnv output now
        lse = analyse.chemenv.get_light_structure_environment(only_cation_environments=True)

        for coord, neighbor_sets in zip(lse.coordination_environments, lse.neighbors_sets):
            # we are only looking at cation-anion bonds here and we only return environments of cations
            if neighbor_sets is not None:
                print(coord[0])
                print(neighbor_sets[0])
                # This returns the list of all neighboring atoms as PeriodicSite objects including coordinates.
                # This should be useful for computing vectors
                print(neighbor_sets[0].neighb_sites)

        # process Lobsterout (similar to outputs from atomate)

        # use lobsterpy for automatic analysis
        # add further infos on COHP

        doc.copy(update=additional_fields)
        return doc


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
