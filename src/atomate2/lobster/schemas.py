"""Module defining amset document schemas."""

import logging
import time
from typing import Any, Dict, List, Tuple, Union, Optional


import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from pymatgen.core import Structure
from pymatgen.io.lobster import Lobsterin, Lobsterout, Icohplist, Charge

import os
from pathlib import Path
from pydantic import BaseModel, Field

from atomate2 import __version__

from atomate2.common.schemas.structure import StructureMetadata
from atomate2.utils.datetime import datetime_str
# from atomate2.common.files import get_zfile
# from atomate2.utils.path import get_uri
# from atomate2.common.schemas.math import Matrix3D, Vector3D


try:
    from lobsterpy.cohp.analyze import Analysis
    from lobsterpy.cohp.describe import Description
except ImportError:
    Analysis = None
    Description = None

__all__ = ["LobsterTaskDocument"]

logger = logging.getLogger(__name__)


class LobsteroutModel(BaseModel):
    """Collection to store computational settings from the LOBSTER computation."""

    restart_from_projection: bool = Field("Bool indicating if the run has been restarted from a projection")
    lobster_version: str = Field("Lobster version")
    threads: int = Field("Number of threads that Lobster ran on")
    dft_program: str = Field("DFT program was used for this run")
    chargespilling: float = Field("Absolute charge spilling")
    totalspilling: float = Field("Total spilling")
    elements: str = Field("Elements in structure")
    basistype: str = Field("Basis set used in Lobster")
    basisfunctions: str = Field("basis_functions")
    timing: dict = Field("Dict with infos on timing")
    warnings: str = Field("Warnings")
    orthonormalization: str = Field("info_orthonormalization")
    infos: str = Field("info_lines")
    hasDOSCAR: bool = Field("Bool indicating if DOSCAR is present.")
    hasCOHPCAR: bool = Field("Bool indicating if COHPCAR is present.")
    hasCOOPCAR: bool = Field("Bool indicating if COOPCAR is present.")
    hasCOBICAR: bool = Field("Bool indicating if COBICAR is present.")
    hasCHARGE: bool = Field("Bool indicating if CHARGE is present.")
    hasmadelung: bool = Field("Bool indicating if Madelung file is present.")
    hasProjection: bool = Field("Bool indicating if projection file is present.")
    hasbandoverlaps: bool = Field("Bool indicating if BANDOVERLAPS file is presetn")
    hasfatband: bool = Field("Bool indicating if Fatbands are present.")
    hasGrossPopulation: bool = Field("Bool indicating if GrossPopulations file is present.")
    hasDensityOfEnergies: bool = Field("Bool indicating if DensityofEnergies is present")


class LobsterinModel(BaseModel):
    """Collection to store input settings for the LOBSTER computation."""

    COHPstartEnergy: float = Field(None, description="Start energy for COHP computation")
    COHPendEnergy: float = Field(None, description="End energy for COHP computation")
    gaussianSmearingWidth: float = Field(None, description="Set the smearing width in eV,default is 0.2 (eV)")
    useDecimalPlaces: int = Field(None, description="Set the decimal places to print in output files, default is 5")
    COHPSteps: int = Field(None, description="Number steps in COHPCAR; similar to NEDOS of VASP")
    basisSet: str = Field(None, description="basis set of computation")
    cohpGenerator: float = Field(None, description="Build the list of atom pairs to be analyzed using given distance")
    saveProjectionToFile: float = Field(None, description="Save the results of projections")
    basisfunctions: str = Field(None, description="Specify the basis functions for element")


class CondensedBondingAnalysis(BaseModel):
    """Collection to store condensed bonding analysis data from LobsterPy based on ICOHP"""

    formula: str = Field(None, description="Pretty formula of the structure")
    max_considered_bond_length: Tuple[float] = Field(None, description="Maximum bond length considered "
                                                                       "in bonding analysis")
    limit_icohp: List[float] = Field(None, description="ICOHP range considered in co-ordination environment analysis")
    number_of_considered_ions: int = Field(None, description="number of ions detected based on Mulliken/Löwdin Charges")
    sites: dict = Field(None, description="Dict object that describes bond summary stats, "
                                          "bonding/antibonding percentage and its coordination environment")
    type_charges: str = Field(None, description="Charge type considered for assinging valences in bonding analysis")
    madelung_energy: float = Field(None, description="Total electrostatic energy for the structure "
                                                     "based on chosen type_charges")
    cutoff_icohp: float = Field(None, description="Percent limiting the ICOHP values to be considered"
                                                  " relative to strongest ICOHP")
    summed_spins: bool = Field(None, description="Bool stating whether to sum spin channels during analysis")
    start: Optional[float] = Field(None, description="Sets the lower limit of energy relative to Fermi for evaluating"
                                                     " bonding/anti-bonding percentages in the bond"
                                                     " if set to None, all energies up-to the Fermi is considered")
    cohp_plot_data: dict[float] = Field(None, description="Stores the COHP plot data based on relevant bond labels "
                                                          "for site as keys")
    run_time: float = Field(None, description="Time needed to run Lobsterpy condensed bonding analysis")


class StrongestBonds(BaseModel):
    """Collection to store strongest bonds extracted from ICOHPLIST/ICOOPLIST/ICOBILIST data from LOBSTER runs"""

    only_cation_anion: bool = Field(None, description="If True, only information of cation-anion pairs "
                                                      "bond strength ,length will be returned ")
    are_coops: bool = Field(None, description="Denotes whether the file consists of ICOOPs")
    are_cobis: bool = Field(None, description="Denotes whether the file consists of ICOBIs")
    strongest_bonds: dict = Field(None,
                                  description="Dict with infos on bond strength, length between cation-anion pairs")


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
    lobsterout: LobsteroutModel = Field("Lobster out data")
    lobsterin: LobsterinModel = Field("Lobster calculation inputs")
    lobsterpy_data: CondensedBondingAnalysis = Field("Model describing the LobsterPy data")
    lobster_strongest_bonds_ICOHP: StrongestBonds = Field("Describes the strongest cation-anion ICOHP bonds")
    lobster_strongest_bonds_ICOOP: StrongestBonds = Field("Describes the strongest cation-anion ICOOP bonds")
    lobster_strongest_bonds_ICOBI: StrongestBonds = Field("Describes the strongest cation-anion ICOBI bonds")
    # COHPData
    # COOPData
    # COBIData

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

        # struct = Structure.from_file(get_zfile(directory_listing=".", base_name="POSCAR"))
        lobsterout_here = Lobsterout("lobsterout.gz")
        lobsterout_doc = lobsterout_here.get_doc()
        lobsterin_here = Lobsterin.from_file(os.path.join(dir_name, "lobsterin"))
        # cation anion-mode
        start = time.time()
        analyse = Analysis(
            path_to_poscar=os.path.join(dir_name, "POSCAR"),
            path_to_icohplist=os.path.join(dir_name, "ICOHPLIST.lobster"),
            path_to_cohpcar=os.path.join(dir_name, "COHPCAR.lobster"),
            path_to_charge=os.path.join(dir_name, "CHARGE.lobster"),
            summed_spins=True,
            cutoff_icohp=0.10
        )

        cba_run_time = time.time() - start

        cba = analyse.condensed_bonding_analysis  # initialize lobsterpy condensed bonding analysis
        cba_cohp_plot_data = {}  # Initialize dict to store plot data
        for site, val in cba.condensed_bonding_analysis['sites'].items():
            cohp_data = cba.chemenv.completecohp.get_summed_cohp_by_label_list(val['relevant_bonds']).as_dict()
            spinup_cohps = cohp_data["COHP"]["1"]
            spindown_cohps = cohp_data["COHP"]["-1"]
            energies = cohp_data["energies"]
            efermi = cohp_data['efermi']
            cba_cohp_plot_data.update({site: {"COHP_spin_up": spinup_cohps,
                                              "COHP_spin_down": spindown_cohps,
                                              "Energies": energies,
                                              "Efermi": efermi}})

        lpca = CondensedBondingAnalysis(formula=cba['formula'],
                                        max_considered_bond_length=cba['max_considered_bond_length'],
                                        limit_icohp=cba['limit_icohp'],
                                        number_of_considered_ions=cba['number_of_considered_ions'],
                                        sites=cba['sites'],
                                        type_charges=cba['type_charges'],
                                        madelung_energy=cba['madelung_energy'],
                                        cohp_plot_data=cba_cohp_plot_data,
                                        run_time=cba_run_time)

        lobsout = LobsteroutModel(**lobsterout_doc)
        lobsin = LobsterinModel(**lobsterin_here)

        ch = Charge(os.path.join(dir_name, "CHARGE.lobster"))
        icohplist = Icohplist(filename=os.path.join(dir_name, "ICOHPLIST.lobster"), are_cobis=False, are_coops=False)
        icobilist = Icohplist(filename=os.path.join(dir_name, "ICOBILIST.lobster"), are_cobis=True, are_coops=False)
        icooplist = Icohplist(filename=os.path.join(dir_name, "ICOOPLIST.lobster"), are_coops=True, are_cobis=False)

        icohp_dict = icohplist.icohpcollection.as_dict()
        icobi_dict = icobilist.icohpcollection.as_dict()
        icoop_dict = icooplist.icohpcollection.as_dict()

        def get_strng_bonds(bondlist, are_cobis, are_coops, ch=ch):
            charges = {}
            for i, v in enumerate(ch.atomlist):
                if ch.Mulliken[i] > 0:
                    charges.update({v.rstrip('0123456789'): 'Cation'})
                else:
                    charges.update({v.rstrip('0123456789'): 'Anion'})

            bonds = []
            icohp_all = []
            lengths = []
            for a, b, c, l in zip(bondlist['list_atom1'], bondlist['list_atom2'],
                                  bondlist['list_icohp'], bondlist['list_length']):
                bonds.append(a.rstrip('0123456789') + '-' + b.rstrip('0123456789'))
                icohp_all.append(sum(c.values()))
                lengths.append(l)

            bond_labels_unique = list(set(bonds))
            sep_blabels = [[] for _ in range(len(bond_labels_unique))]
            sep_icohp = [[] for _ in range(len(bond_labels_unique))]
            sep_lengths = [[] for _ in range(len(bond_labels_unique))]

            for i, val in enumerate(bond_labels_unique):
                for j, val2 in enumerate(bonds):
                    if val == val2:
                        sep_blabels[i].append(val2)
                        sep_icohp[i].append(icohp_all[j])
                        sep_lengths[i].append(lengths[j])
            if not are_cobis and not are_coops:
                bond_dict = {}
                for i, lab in enumerate(bond_labels_unique):
                    label = lab.split('-')
                    if label[0] != label[1]:
                        if charges[label[0]] != charges[label[1]]:
                            index = np.argmin(sep_icohp[i])
                            bond_dict.update({lab: {'ICOHP': min(sep_icohp[i]),
                                                    'length': sep_lengths[i][index]}})
                return bond_dict

            if are_cobis and not are_coops:
                bond_dict = {}
                for i, lab in enumerate(bond_labels_unique):
                    label = lab.split('-')
                    if label[0] != label[1]:
                        if charges[label[0]] != charges[label[1]]:
                            index = np.argmin(sep_icohp[i])
                            bond_dict.update({lab: {'ICOBI': max(sep_icohp[i]),
                                                    'length': sep_lengths[i][index]}})
                return bond_dict

            if not are_cobis and are_coops:
                bond_dict = {}
                for i, lab in enumerate(bond_labels_unique):
                    label = lab.split('-')
                    if label[0] != label[1]:
                        if charges[label[0]] != charges[label[1]]:
                            index = np.argmin(sep_icohp[i])
                            bond_dict.update({lab: {'ICOOP': max(sep_icohp[i]),
                                                    'length': sep_lengths[i][index]}})
                return bond_dict

        icohp_bond_dict = get_strng_bonds(icohp_dict, are_cobis=False, are_coops=False)
        icoop_bond_dict = get_strng_bonds(icoop_dict, are_cobis=False, are_coops=True)
        icobi_bond_dict = get_strng_bonds(icobi_dict, are_cobis=True, are_coops=False)

        sb_ichop = StrongestBonds(are_cobis=False, are_coops=False, strongest_bonds=icohp_bond_dict)
        sb_icoop = StrongestBonds(are_cobis=False, are_coops=True, strongest_bonds=icoop_bond_dict)
        sb_icobi = StrongestBonds(are_cobis=True, are_coops=False, strongest_bonds=icobi_bond_dict)

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

        # doc.copy(update=additional_fields)
        return cls(
            lobsterin=lobsin,
            lobsterout=lobsout,
            lobsterpy_data=lpca,
            lobster_strongest_bonds_ICOHP=sb_ichop,
            lobster_strongest_bonds_ICOOP=sb_icoop,
            lobster_strongest_bonds_ICOBI=sb_icobi,
        )  # doc


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
