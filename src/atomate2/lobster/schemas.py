"""Module defining amset document schemas."""

import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
from monty.dev import requires
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.dos import LobsterCompleteDos
from pymatgen.io.lobster import (
    Charge,
    Doscar,
    Icohplist,
    Lobsterin,
    Lobsterout,
    MadelungEnergies,
)

from atomate2 import __version__
from atomate2.utils.datetime import datetime_str

try:
    from lobsterpy.cohp.analyze import Analysis
    from lobsterpy.cohp.describe import Description
except ImportError:
    Analysis = None
    Description = None

__all__ = ["LobsterTaskDocument"]

logger = logging.getLogger(__name__)


class LobsteroutModel(BaseModel):
    """Definition of computational settings from the LOBSTER computation."""

    restart_from_projection: bool = Field(
        None,
        description="Bool indicating if the run has been restarted from a projection",
    )
    lobster_version: str = Field(None, description="Lobster version")
    threads: int = Field(None, description="Number of threads that Lobster ran on")
    dft_program: str = Field(None, description="DFT program was used for this run")
    charge_spilling: list = Field(None, description="Absolute charge spilling")
    total_spilling: list = Field(None, description="Total spilling")
    elements: list = Field(None, description="Elements in structure")
    basis_type: list = Field(None, description="Basis set used in Lobster")
    basis_functions: list = Field(None, description="basis_functions")
    timing: Any = Field(None, description="Dict with infos on timing")
    warning_lines: list = Field(None, description="Warnings")
    info_orthonormalization: list = Field(
        None, description="additional information on orthonormalization"
    )
    info_lines: list = Field(
        None, description="list of strings with additional info lines"
    )
    has_doscar: bool = Field(None, description="Bool indicating if DOSCAR is present.")
    has_cohpcar: bool = Field(
        None, description="Bool indicating if COHPCAR is present."
    )
    has_coopcar: bool = Field(
        None, description="Bool indicating if COOPCAR is present."
    )
    has_cobicar: bool = Field(
        None, description="Bool indicating if COBICAR is present."
    )
    has_charge: bool = Field(None, description="Bool indicating if CHARGE is present.")
    has_madelung: bool = Field(
        None, description="Bool indicating if Madelung file is present."
    )
    has_projection: bool = Field(
        None, description="Bool indicating if projection file is present."
    )
    has_bandoverlaps: bool = Field(
        None, description="Bool indicating if BANDOVERLAPS file is presetn"
    )
    has_fatbands: bool = Field(
        None, description="Bool indicating if Fatbands are present."
    )
    has_grosspopulation: bool = Field(
        None, description="Bool indicating if GrossPopulations file is present."
    )
    has_density_of_energies: bool = Field(
        None, description="Bool indicating if DensityofEnergies is present"
    )


class LobsterinModel(BaseModel):
    """Definition of input settings for the LOBSTER computation."""

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
    """Definition of condensed bonding analysis data from LobsterPy ICOHP."""

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
        description="Charge type considered for assigning valences in bonding analysis",
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
        description="Specifies types of bond considered in LobsterPy analysis",
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

    @classmethod
    def from_directory(
        cls,
        dir_name: Union[str, Path],
        save_cohp_plots: bool = True,
        plot_kwargs: dict = None,
    ):
        """
        Create a task document from a directory containing LOBSTER files.

        Parameters
        ----------
        dir_name : path or str
            The path to the folder containing the calculation outputs.
        save_cohp_plots : bool
            Bool to indicate whether automatic cohp plots and jsons
            from lobsterpy will be generated.
        plot_kwargs : dict
            kwargs to change plotting options in lobsterpy.
        """
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        dir_name = Path(dir_name)

        cohpcar_path = dir_name / "COHPCAR.lobster.gz"
        charge_path = dir_name / "CHARGE.lobster.gz"
        structure_path = dir_name / "POSCAR.gz"
        icohplist_path = dir_name / "ICOHPLIST.lobster.gz"
        icobilist_path = dir_name / "ICOBILIST.lobster.gz"
        icooplist_path = dir_name / "ICOOPLIST.lobster.gz"

        try:
            # cation anion-mode
            start = time.time()
            analyse = Analysis(
                path_to_poscar=structure_path,
                path_to_icohplist=icohplist_path,
                path_to_cohpcar=cohpcar_path,
                path_to_charge=charge_path,
                summed_spins=True,
                cutoff_icohp=0.10,
                whichbonds="cation-anion",
            )
            cba_run_time = time.time() - start
        except ValueError:
            # all bonds
            start = time.time()
            analyse = Analysis(
                path_to_poscar=structure_path,
                path_to_icohplist=icohplist_path,
                path_to_cohpcar=cohpcar_path,
                path_to_charge=charge_path,
                summed_spins=True,
                cutoff_icohp=0.10,
                whichbonds="all",
            )
            cba_run_time = time.time() - start

        # initialize lobsterpy condensed bonding analysis
        cba = analyse.condensed_bonding_analysis
        cba_cohp_plot_data = {}  # Initialize dict to store plot data

        set_cohps = analyse.set_cohps
        set_labels_cohps = analyse.set_labels_cohps
        set_inequivalent_cations = analyse.set_inequivalent_ions
        struct = analyse.structure
        for _iplot, (ication, labels, cohps) in enumerate(
            zip(set_inequivalent_cations, set_labels_cohps, set_cohps)
        ):
            label_str = f"{str(struct[ication].specie)}{str(ication + 1)}: "
            for label, cohp in zip(labels, cohps):
                if label is not None:
                    cba_cohp_plot_data.update(
                        {
                            label_str
                            + label: {
                                "COHP": list(cohp.get_cohp()[Spin.up]),
                                "ICOHP": list(cohp.get_icohp()[Spin.up]),
                                "Energies": list(cohp.energies),
                                "Efermi": cohp.efermi,
                            }
                        }
                    )

        describe = Description(analysis_object=analyse)

        condensed_bonding_analysis = CondensedBondingAnalysis(
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
        if save_cohp_plots:
            describe.plot_cohps(
                save=True,
                filename="automatic_cohp_plots.pdf",
                skip_show=True,
                **plot_kwargs,
            )
            import json

            with open(dir_name / "condensed_bonding_analysis.json", "w") as fp:
                json.dump(analyse.condensed_bonding_analysis, fp)
            with open(dir_name / "condensed_bonding_analysis.txt", "w") as fp:
                for line in describe.text:
                    fp.write(line + "\n")

        # Read in strongest icohp values
        sb_icobi, sb_icohp, sb_icoop = _identify_strongest_bonds(
            analyse, icobilist_path, icohplist_path, icooplist_path
        )
        return (
            condensed_bonding_analysis,
            struct,
            describe,
            sb_icobi,
            sb_icohp,
            sb_icoop,
        )


class StrongestBonds(BaseModel):
    """Strongest bonds extracted from ICOHPLIST/ICOOPLIST/ICOBILIST from LOBSTER."""

    which_bonds: str = Field(
        None,
        description="Denotes whether the information "
        "is for cation-anion pairs or all bonds",
    )
    are_coops: bool = Field(
        None, description="Denotes whether the file consists of ICOOPs"
    )
    are_cobis: bool = Field(
        None, description="Denotes whether the file consists of ICOBIs"
    )
    strongest_bonds: dict = Field(
        None,
        description="Dict with infos on bond strength,"
        " length between cation-anion pairs",
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
        description="Atomic charges dict from LOBSTER"
        " based on Mulliken and Loewdin charge analysis",
    )
    lobsterout: LobsteroutModel = Field(None, description="Lobster out data")
    lobsterin: LobsterinModel = Field(None, description="Lobster calculation inputs")
    lobsterpy_data: CondensedBondingAnalysis = Field(
        None, description="Model describing the LobsterPy data"
    )
    lobsterpy_summary_text: str = Field(
        None,
        description="Stores LobsterPy automatic analysis summary text",
    )
    strongest_bonds_icohp: StrongestBonds = Field(
        None, description="Describes the strongest cation-anion ICOHP bonds"
    )
    strongest_bonds_icoop: StrongestBonds = Field(
        None, description="Describes the strongest cation-anion ICOOP bonds"
    )
    strongest_bonds_icobi: StrongestBonds = Field(
        None, description="Describes the strongest cation-anion ICOBI bonds"
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
        description="Madelung energies dict from"
        " LOBSTER based on Mulliken and Loewdin charges",
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
        additional_fields: dict = None,
        store_lso_dos: bool = False,
        save_cohp_plots: bool = True,
        plot_kwargs: dict = None,
    ):
        """
        Create a task document from a directory containing LOBSTER files.

        Parameters
        ----------
        dir_name : path or str
            The path to the folder containing the calculation outputs.
        additional_fields : dict
            Dictionary of additional fields to add to output document.
        store_lso_dos : bool
            Whether to store the LSO DOS.
        save_cohp_plots : bool
            Bool to indicate whether automatic cohp plots and jsons
            from lobsterpy will be generated.
        plot_kwargs : dict
            kwargs to change plotting options in lobsterpy.

        Returns
        -------
        LobsterTaskDocument
            A task document for the lobster calculation.
        """
        additional_fields = {} if additional_fields is None else additional_fields
        dir_name = Path(dir_name)

        # Read in lobsterout and lobsterin
        lobsterout_doc = Lobsterout(dir_name / "lobsterout.gz").get_doc()
        lobster_out = LobsteroutModel(**lobsterout_doc)
        lobster_in = LobsterinModel(**Lobsterin.from_file(dir_name / "lobsterin.gz"))

        icohplist_path = dir_name / "ICOHPLIST.lobster.gz"
        cohpcar_path = dir_name / "COHPCAR.lobster.gz"
        cobicar_path = dir_name / "COBICAR.lobster.gz"
        coopcar_path = dir_name / "COOPCAR.lobster.gz"
        charge_path = dir_name / "CHARGE.lobster.gz"
        doscar_path = dir_name / "DOSCAR.lobster.gz"
        structure_path = dir_name / "POSCAR.gz"
        madelung_energies_path = dir_name / "MadelungEnergies.lobster.gz"

        # Do automatic bonding analysis with LobsterPy
        condensed_bonding_analysis_data = None
        sb_icobi = None
        sb_icohp = None
        sb_icoop = None
        struct = None
        describe = None
        if icohplist_path.exists() and cohpcar_path.exists() and charge_path.exists():
            (
                condensed_bonding_analysis,
                struct,
                describe,
                sb_icobi,
                sb_icohp,
                sb_icoop,
            ) = CondensedBondingAnalysis.from_directory(
                dir_name, save_cohp_plots=save_cohp_plots, plot_kwargs=plot_kwargs
            )

        # Read in charges
        charges = None
        if charge_path.exists():
            charge = Charge(charge_path)
            charges = {"Mulliken": charge.Mulliken, "Loewdin": charge.Loewdin}

        # Read in COHP, COBI, COOP plots
        cohp_obj = None
        if cohpcar_path.exists():
            cohp_obj = CompleteCohp.from_file(
                fmt="LOBSTER",
                structure_file=structure_path,
                filename=cohpcar_path,
                are_coops=False,
                are_cobis=False,
            )

        coop_obj = None
        if coopcar_path.exists():
            coop_obj = CompleteCohp.from_file(
                fmt="LOBSTER",
                structure_file=structure_path,
                filename=coopcar_path,
                are_coops=True,
                are_cobis=False,
            )

        cobi_obj = None
        if cobicar_path.exists():
            cobi_obj = CompleteCohp.from_file(
                fmt="LOBSTER",
                structure_file=structure_path,
                filename=cobicar_path,
                are_coops=False,
                are_cobis=True,
            )

        # Read in DOS
        dos = None
        if doscar_path.exists():
            doscar_lobster = Doscar(
                doscar="DOSCAR.lobster.gz", structure_file="POSCAR.gz"
            )
            dos = doscar_lobster.completedos

        # Read in LSO DOS
        lso_dos = None
        doscar_lso_path = dir_name / "DOSCAR.LSO.lobster"
        if store_lso_dos and doscar_lso_path.exists():
            doscar_lso_lobster = Doscar(
                doscar=doscar_lso_path, structure_file=structure_path
            )
            lso_dos = doscar_lso_lobster.completedos

        # Read in Madelung energies
        madelung_energies = None
        if madelung_energies_path.exists():
            madelung_obj = MadelungEnergies(filename=madelung_energies_path)

            madelung_energies = {
                "Mulliken": madelung_obj.madelungenergies_Mulliken,
                "Loewdin": madelung_obj.madelungenergies_Loewdin,
                "Ewald_splitting": madelung_obj.ewald_splitting,
            }

        doc = cls(
            structure=struct,
            dir_name=dir_name,
            lobsterin=lobster_in,
            lobsterout=lobster_out,
            lobsterpy_data=condensed_bonding_analysis_data,
            lobsterpy_summary_text=" ".join(describe.text),
            strongest_bonds_icohp=sb_icohp,
            strongest_bonds_icoop=sb_icoop,
            strongest_bonds_icobi=sb_icobi,
            cohp_data=cohp_obj,
            coop_data=coop_obj,
            cobi_data=cobi_obj,
            dos=dos,
            lso_dos=lso_dos,
            charges=charges,
            madelung_energies=madelung_energies,
        )
        doc = doc.copy(update=additional_fields)
        return doc


def _identify_strongest_bonds(
    analyse: Analysis,
    icobilist_path: Path,
    icohplist_path: Path,
    icooplist_path: Path,
):
    """

    Parameters
    ----------
    analyse : .Analysis
        Analysis object from lobsterpy automatic analysis
    icobilist_path : Path or str
        Path to ICOBILIST.lobster
    icohplist_path : Path or str
        Path to ICOHPLIST.lobster
    icooplist_path : Path or str
        Path to ICOOPLIST.lobster

    Returns
    -------
    Tuple[StrongestBonds]
        Tuple of StrongestBonds
    """
    data = [
        (icohplist_path, False, False),
        (icobilist_path, True, False),
        (icooplist_path, False, True),
    ]
    output = []
    for file, are_cobis, are_coops in data:
        if file.exists():
            icohplist = Icohplist(
                filename=icohplist_path,
                are_cobis=are_cobis,
                are_coops=are_coops,
            )
            bond_dict = _get_strong_bonds(
                icohplist.icohpcollection.as_dict(),
                relevant_bonds=analyse.final_dict_bonds,
                are_cobis=are_cobis,
                are_coops=are_coops,
            )
            output.append(
                StrongestBonds(
                    are_cobis=are_cobis,
                    are_coops=are_coops,
                    strongest_bonds=bond_dict,
                    which_bonds=analyse.whichbonds,
                )
            )
        else:
            output.append(None)
    return output


def _get_strong_bonds(
    bondlist: dict, are_cobis: bool, are_coops: bool, relevant_bonds: dict
):
    """
    Identify the strongest bonds from a list of bonds.

    Parameters
    ----------
    bondlist : dict
        dict including bonding information
    are_cobis : bool
        True if these are cobis
    are_coops : bool
        True if these are coops
    relevant_bonds : dict
        Dict include all bonds that are considered.

    Returns
    -------
    dict
        Dictionary including strongest bonds.
    """
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
    sep_blabels: List[List[str]] = [[] for _ in range(len(bond_labels_unique))]
    sep_icohp: List[List[float]] = [[] for _ in range(len(bond_labels_unique))]
    sep_lengths: List[List[float]] = [[] for _ in range(len(bond_labels_unique))]

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
            for rel_bnd in relevant_bonds:
                rel_bnd_list = rel_bnd.split("-")
                rel_bnd_list.sort()
                if label == rel_bnd_list:
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
            for rel_bnd in relevant_bonds:
                rel_bnd_list = rel_bnd.split("-")
                rel_bnd_list.sort()
                if label == rel_bnd_list:
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
            for rel_bnd in relevant_bonds:
                rel_bnd_list = rel_bnd.split("-")
                rel_bnd_list.sort()
                if label == rel_bnd_list:
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
