"""Module defining amset document schemas."""

import logging
import time
from typing import Any, Union, Optional, Dict
import os
from pathlib import Path
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from pymatgen.core import Structure
from pymatgen.io.lobster import (
    Lobsterin,
    Lobsterout,
    Icohplist,
    Charge,
    MadelungEnergies,
)
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.io.lobster import Doscar
from pymatgen.electronic_structure.dos import LobsterCompleteDos

from pydantic import BaseModel, Field

from atomate2 import __version__

# from atomate2.common.schemas.structure import StructureMetadata
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

    restart_from_projection: bool = Field(
        "Bool indicating if the run has been restarted from a projection"
    )
    lobster_version: str = Field("Lobster version")
    threads: int = Field("Number of threads that Lobster ran on")
    dft_program: str = Field("DFT program was used for this run")
    charge_spilling: list = Field("Absolute charge spilling")
    total_spilling: list = Field("Total spilling")
    elements: list = Field("Elements in structure")
    basis_type: list = Field("Basis set used in Lobster")
    basis_functions: list = Field("basis_functions")
    timing: Any = Field("Dict with infos on timing")
    warning_lines: list = Field("Warnings")
    info_orthonormalization: list = Field("info_orthonormalization")
    info_lines: list = Field("info_lines")
    has_doscar: bool = Field("Bool indicating if DOSCAR is present.")
    has_cohpcar: bool = Field("Bool indicating if COHPCAR is present.")
    has_coopcar: bool = Field("Bool indicating if COOPCAR is present.")
    has_cobicar: bool = Field("Bool indicating if COBICAR is present.")
    has_charge: bool = Field("Bool indicating if CHARGE is present.")
    has_madelung: bool = Field("Bool indicating if Madelung file is present.")
    has_projection: bool = Field("Bool indicating if projection file is present.")
    has_bandoverlaps: bool = Field("Bool indicating if BANDOVERLAPS file is presetn")
    has_fatbands: bool = Field("Bool indicating if Fatbands are present.")
    has_grosspopulation: bool = Field(
        "Bool indicating if GrossPopulations file is present."
    )
    has_density_of_energies: bool = Field(
        "Bool indicating if DensityofEnergies is present"
    )


class LobsterinModel(BaseModel):
    """Collection to store input settings for the LOBSTER computation."""

    cohpstartenergy: float = Field(
        None, description="Start energy for COHP computation"
    )
    cohpendenergy: float = Field(None, description="End energy for COHP computation")

    gaussiansmearingwidth: float = Field(
        None, description="Set the smearing width in eV,default is 0.2 (eV)"
    )
    usedecimalplaces: int = Field(
        None,
        description="Set the decimal places to print in output files, default is 5",
    )
    cohpsteps: float = Field(
        None, description="Number steps in COHPCAR; similar to NEDOS of VASP"
    )
    basisset: str = Field(None, description="basis set of computation")
    cohpgenerator: str = Field(
        None,
        description="Build the list of atom pairs to be analyzed using given distance",
    )
    saveprojectiontofile: bool = Field(
        None, description="Save the results of projections"
    )
    lsodos: bool = Field(
        None, description="Writes DOS output from the orthonormalized LCAO basis"
    )
    basisfunctions: list = Field(
        None, description="Specify the basis functions for element"
    )


class CondensedBondingAnalysis(BaseModel):
    """Collection to store condensed bonding analysis data from LobsterPy based on ICOHP"""

    formula: str = Field(None, description="Pretty formula of the structure")
    max_considered_bond_length: Any = Field(
        None, description="Maximum bond length considered " "in bonding analysis"
    )
    limit_icohp: list = Field(
        None, description="ICOHP range considered in co-ordination environment analysis"
    )
    number_of_considered_ions: int = Field(
        None, description="number of ions detected based on Mulliken/L�wdin Charges"
    )
    sites: dict = Field(
        None,
        description="Dict object that describes bond summary stats, "
        "bonding/antibonding percentage and its coordination environment",
    )
    type_charges: str = Field(
        None,
        description="Charge type considered for assinging valences in bonding analysis",
    )
    cutoff_icohp: float = Field(
        None,
        description="Percent limiting the ICOHP values to be considered"
        " relative to strongest ICOHP",
    )
    summed_spins: bool = Field(
        None, description="Bool stating whether to sum spin channels during analysis"
    )
    start: Optional[float] = Field(
        None,
        description="Sets the lower limit of energy relative to Fermi for evaluating"
        " bonding/anti-bonding percentages in the bond"
        " if set to None, all energies up-to the Fermi is considered",
    )
    cohp_plot_data: dict = Field(
        None,
        description="Stores the COHP plot data based on relevant bond labels "
        "for site as keys",
    )
    which_bonds: str = Field(
        None,
        description="Specifies types of bond considerd in LobsterPy analysis",
    )
    final_dict_bonds: dict = Field(
        None,
        description="Dict consisting information on ICOHPs per bond type",
    )
    final_dict_ions: dict = Field(
        None,
        description="Dict consisting information on environments of cations",
    )
    run_time: float = Field(
        None, description="Time needed to run Lobsterpy condensed bonding analysis"
    )


class StrongestBonds(BaseModel):
    """Collection to store strongest bonds extracted from ICOHPLIST/ICOOPLIST/ICOBILIST data from LOBSTER runs"""

    which_bonds: str = Field(
        None,
        description="Denotes whether the information is for cation-anion pairs or all bonds",
    )
    are_coops: bool = Field(
        None, description="Denotes whether the file consists of ICOOPs"
    )
    are_cobis: bool = Field(
        None, description="Denotes whether the file consists of ICOBIs"
    )
    strongest_bonds: dict = Field(
        None,
        description="Dict with infos on bond strength, length between cation-anion pairs",
    )


class LobsterTaskDocument(BaseModel):
    """Definition of LOBSTER task document."""

    structure: Structure = Field(None, description="The structure used in this task")
    dir_name: Any = Field(None, description="The directory for this Lobster task")
    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp for this task document was last updated",
    )
    charges: dict = Field(
        None,
        description="Atomic charges dict from LOBSTER based on Mulliken and Loewdin charge analysis",
    )
    lobsterout: LobsteroutModel = Field("Lobster out data")
    lobsterin: LobsterinModel = Field("Lobster calculation inputs")
    lobsterpy_data: CondensedBondingAnalysis = Field(
        "Model describing the LobsterPy data"
    )
    lobsterpy_summary_text: str = Field(
        None,
        description="Stores LobsterPy automatic analysis summary text",
    )
    strongest_bonds_icohp: StrongestBonds = Field(
        "Describes the strongest cation-anion ICOHP bonds"
    )
    strongest_bonds_icoop: StrongestBonds = Field(
        "Describes the strongest cation-anion ICOOP bonds"
    )
    strongest_bonds_icobi: StrongestBonds = Field(
        "Describes the strongest cation-anion ICOBI bonds"
    )
    cohp_data: CompleteCohp = Field(
        None, description="pymatgen CompleteCohp object with COHP data"
    )
    coop_data: CompleteCohp = Field(
        None, description="pymatgen CompleteCohp object with COOP data"
    )
    cobi_data: CompleteCohp = Field(
        None, description="pymatgen CompleteCohp object with COBI data"
    )
    dos: LobsterCompleteDos = Field(
        None, description="pymatgen pymatgen.io.lobster.Doscar.completedos data"
    )
    lso_dos: LobsterCompleteDos = Field(
        None, description="pymatgen pymatgen.io.lobster.Doscar.completedos data"
    )
    madelung_energies: dict = Field(
        None,
        description="Madelung energies dict from LOBSTER based on Mulliken and Loewdin charges",
    )

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
        additional_fields: list[str]|None = None
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

        additional_fields = [] if additional_fields is None else additional_fields
        dir_name = Path(dir_name)
        # do automatic analysis with lobsterpy and provide data

        struct = Structure.from_file(os.path.join(dir_name, "POSCAR.gz"))
        lobsterout_here = Lobsterout("lobsterout.gz")
        lobsterout_doc = lobsterout_here.get_doc()
        lobsterin_here = Lobsterin.from_file(os.path.join(dir_name, "lobsterin.gz"))
        # cation anion-mode

        try:
            start = time.time()
            analyse = Analysis(
                path_to_poscar=os.path.join(dir_name, "POSCAR.gz"),
                path_to_icohplist=os.path.join(dir_name, "ICOHPLIST.lobster.gz"),
                path_to_cohpcar=os.path.join(dir_name, "COHPCAR.lobster.gz"),
                path_to_charge=os.path.join(dir_name, "CHARGE.lobster.gz"),
                summed_spins=True,
                cutoff_icohp=0.10,
                whichbonds="cation-anion",
            )

            cba_run_time = time.time() - start
        except ValueError:
            start = time.time()
            analyse = Analysis(
                path_to_poscar=os.path.join(dir_name, "POSCAR.gz"),
                path_to_icohplist=os.path.join(dir_name, "ICOHPLIST.lobster.gz"),
                path_to_cohpcar=os.path.join(dir_name, "COHPCAR.lobster.gz"),
                path_to_charge=os.path.join(dir_name, "CHARGE.lobster.gz"),
                summed_spins=True,
                cutoff_icohp=0.10,
                whichbonds="all",
            )

            cba_run_time = time.time() - start

        cba = (
            analyse.condensed_bonding_analysis
        )  # initialize lobsterpy condensed bonding analysis
        cba_cohp_plot_data = {}  # Initialize dict to store plot data
        for index, (_site, val) in enumerate(
            analyse.condensed_bonding_analysis["sites"].items()
        ):
            cohp_data = analyse.chemenv.completecohp.get_summed_cohp_by_label_list(
                val["relevant_bonds"]
            ).as_dict()
            spinup_cohps = cohp_data["COHP"]["1"]
            spindown_cohps = cohp_data["COHP"]["-1"]
            energies = cohp_data["energies"]
            efermi = cohp_data["efermi"]
            cba_cohp_plot_data.update(
                {
                    analyse.set_labels_cohps[index][0]: {
                        "COHP_spin_up": spinup_cohps,
                        "COHP_spin_down": spindown_cohps,
                        "Energies": energies,
                        "Efermi": efermi,
                    }
                }
            )
        describe = Description(analysis_object=analyse)

        lpca = CondensedBondingAnalysis(
            formula=cba["formula"],
            max_considered_bond_length=cba["max_considered_bond_length"],
            limit_icohp=cba["limit_icohp"],
            number_of_considered_ions=cba["number_of_considered_ions"],
            sites=cba["sites"],
            type_charges=analyse.type_charge,
            cohp_plot_data=cba_cohp_plot_data,
            cutoff_icohp=analyse.cutoff_icohp,
            summed_spins=True,
            which_bonds=analyse.whichbonds,
            final_dict_bonds=analyse.final_dict_bonds,
            final_dict_ions=analyse.final_dict_ions,
            run_time=cba_run_time,
        )

        lobster_out = LobsteroutModel(**lobsterout_doc)
        lobster_in = LobsterinModel(**lobsterin_here)

        charge = Charge(os.path.join(dir_name, "CHARGE.lobster.gz"))

        charges = {"Mulliken": charge.Mulliken, "Loewdin": charge.Loewdin}

        icohplist = Icohplist(
            filename=os.path.join(dir_name, "ICOHPLIST.lobster.gz"),
            are_cobis=False,
            are_coops=False,
        )
        icobilist = Icohplist(
            filename=os.path.join(dir_name, "ICOBILIST.lobster.gz"),
            are_cobis=True,
            are_coops=False,
        )
        icooplist = Icohplist(
            filename=os.path.join(dir_name, "ICOOPLIST.lobster.gz"),
            are_coops=True,
            are_cobis=False,
        )

        icohp_dict = icohplist.icohpcollection.as_dict()
        icobi_dict = icobilist.icohpcollection.as_dict()
        icoop_dict = icooplist.icohpcollection.as_dict()

        def get_strng_bonds(
            bondlist, are_cobis, are_coops, relevant_bonds: dict
        ):
            bonds = []
            icohp_all = []
            lengths = []
            for a, b, c, l in zip(
                bondlist["list_atom1"],
                bondlist["list_atom2"],
                bondlist["list_icohp"],
                bondlist["list_length"],
            ):
                bonds.append(a.rstrip("0123456789") + "-" + b.rstrip("0123456789"))
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
                    label = lab.split("-")
                    label.sort()
                    for rel_bnd in relevant_bonds.keys():
                        rel_bnd_list=rel_bnd.split('-')
                        rel_bnd_list.sort()
                        if label==rel_bnd_list:
                            index = np.argmin(sep_icohp[i])
                            bond_dict.update(
                                {
                                    rel_bnd: {
                                        "ICOHP": min(sep_icohp[i]),
                                        "length": sep_lengths[i][index],
                                    }
                                }
                            )
                return bond_dict

            if are_cobis and not are_coops:
                bond_dict = {}
                for i, lab in enumerate(bond_labels_unique):
                    label = lab.split("-")
                    label.sort()
                    for rel_bnd in relevant_bonds.keys():
                        rel_bnd_list=rel_bnd.split('-')
                        rel_bnd_list.sort()
                        if label==rel_bnd_list:
                            index = np.argmax(sep_icohp[i])
                            bond_dict.update(
                                {
                                    rel_bnd: {
                                        "ICOBI": max(sep_icohp[i]),
                                        "length": sep_lengths[i][index],
                                    }
                                }
                            )
                return bond_dict

            if not are_cobis and are_coops:
                bond_dict = {}
                for i, lab in enumerate(bond_labels_unique):
                    label = lab.split("-")
                    label.sort()
                    for rel_bnd in relevant_bonds.keys():
                        rel_bnd_list=rel_bnd.split('-')
                        rel_bnd_list.sort()
                        if label==rel_bnd_list:
                            index = np.argmax(sep_icohp[i])
                            bond_dict.update(
                                {
                                    rel_bnd: {
                                        "ICOOP": max(sep_icohp[i]),
                                        "length": sep_lengths[i][index],
                                    }
                                }
                            )
                return bond_dict

        icohp_bond_dict = get_strng_bonds(
            icohp_dict,
            are_cobis=False,
            are_coops=False,
            relevant_bonds=analyse.final_dict_bonds,
        )
        icoop_bond_dict = get_strng_bonds(
            icoop_dict,
            are_cobis=False,
            are_coops=True,
            relevant_bonds=analyse.final_dict_bonds,
        )
        icobi_bond_dict = get_strng_bonds(
            icobi_dict,
            are_cobis=True,
            are_coops=False,
            relevant_bonds=analyse.final_dict_bonds,
        )

        sb_ichop = StrongestBonds(
            are_cobis=False,
            are_coops=False,
            strongest_bonds=icohp_bond_dict,
            which_bonds=analyse.whichbonds,
        )
        sb_icoop = StrongestBonds(
            are_cobis=False,
            are_coops=True,
            strongest_bonds=icoop_bond_dict,
            which_bonds=analyse.whichbonds,
        )
        sb_icobi = StrongestBonds(
            are_cobis=True,
            are_coops=False,
            strongest_bonds=icobi_bond_dict,
            which_bonds=analyse.whichbonds,
        )

        cohp_obj = CompleteCohp.from_file(
            fmt="LOBSTER",
            structure_file="POSCAR.gz",
            filename="COHPCAR.lobster.gz",
            are_coops=False,
            are_cobis=False,
        )

        coop_obj = CompleteCohp.from_file(
            fmt="LOBSTER",
            structure_file="POSCAR.gz",
            filename="COOPCAR.lobster.gz",
            are_coops=True,
            are_cobis=False,
        )

        cobi_obj = CompleteCohp.from_file(
            fmt="LOBSTER",
            structure_file="POSCAR.gz",
            filename="COBICAR.lobster.gz",
            are_coops=False,
            are_cobis=True,
        )
        doscar_lobster = Doscar(doscar="DOSCAR.lobster.gz", structure_file="POSCAR.gz")
        dos = doscar_lobster.completedos

        if additional_fields:
            if 'DOSCAR.LSO.lobster' in additional_fields:
                doscar_lso_lobster = Doscar(doscar="DOSCAR.LSO.lobster.gz", structure_file="POSCAR.gz")
                lso_dos = doscar_lso_lobster.completedos
            else:
                lso_dos = None

        madelung_obj = MadelungEnergies(filename="MadelungEnergies.lobster.gz")

        madelung_energies = {
            "Mulliken": madelung_obj.madelungenergies_Mulliken,
            "Loewdin": madelung_obj.madelungenergies_Loewdin,
            "Ewald_splitting": madelung_obj.ewald_splitting,
        }

        return cls(
            structure=struct,
            dir_name=dir_name,
            lobsterin=lobster_in,
            lobsterout=lobster_out,
            lobsterpy_data=lpca,
            lobsterpy_summary_text=" ".join(describe.text),
            strongest_bonds_icohp=sb_ichop,
            strongest_bonds_icoop=sb_icoop,
            strongest_bonds_icobi=sb_icobi,
            cohp_data=cohp_obj,
            coop_data=coop_obj,
            cobi_data=cobi_obj,
            dos=dos,
            lso_dos=lso_dos,
            charges=charges,
            madelung_energies=madelung_energies,
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
