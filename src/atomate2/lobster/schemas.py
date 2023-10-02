"""Module defining lobster document schemas."""

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Union

import ijson
import numpy as np
from emmet.core.structure import StructureMetadata
from monty.dev import requires
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.electronic_structure.cohp import Cohp, CompleteCohp
from pymatgen.electronic_structure.dos import LobsterCompleteDos
from pymatgen.io.lobster import (
    Bandoverlaps,
    Charge,
    Doscar,
    Grosspop,
    Icohplist,
    Lobsterin,
    Lobsterout,
    MadelungEnergies,
    SitePotential,
)

from atomate2 import __version__
from atomate2.utils.datetime import datetime_str

try:
    from lobsterpy.cohp.analyze import Analysis
    from lobsterpy.cohp.describe import Description
except ImportError:
    Analysis = None
    Description = None

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
    has_doscar_lso: bool = Field(
        None, description="Bool indicating if DOSCAR.LSO is present."
    )
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
        None,
        description="Bool indicating if Site Potentials and Madelung file is present.",
    )
    has_projection: bool = Field(
        None, description="Bool indicating if projection file is present."
    )
    has_bandoverlaps: bool = Field(
        None, description="Bool indicating if BANDOVERLAPS file is present"
    )
    has_fatbands: bool = Field(
        None, description="Bool indicating if Fatbands are present."
    )
    has_grosspopulation: bool = Field(
        None, description="Bool indicating if GROSSPOP file is present."
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
        None, description="Maximum bond length considered in bonding analysis"
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
        None,
        description="Bool that states if the spin channels in the "
        "cohp_plot_data are summed.",
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
        which_bonds: str = "all",
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
        which_bonds: str
            mode for condensed bonding analysis: "cation-anion" and "all".
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
            start = time.time()
            analyse = Analysis(
                path_to_poscar=structure_path,
                path_to_icohplist=icohplist_path,
                path_to_cohpcar=cohpcar_path,
                path_to_charge=charge_path,
                summed_spins=False,  # we will always use spin polarization here
                cutoff_icohp=0.10,
                whichbonds=which_bonds,
            )
            cba_run_time = time.time() - start
            # initialize lobsterpy condensed bonding analysis
            cba = analyse.condensed_bonding_analysis

            cba_cohp_plot_data = {}  # Initialize dict to store plot data

            seq_cohps = analyse.seq_cohps
            seq_labels_cohps = analyse.seq_labels_cohps
            seq_ineq_cations = analyse.seq_ineq_ions
            struct = analyse.structure

            for _iplot, (ication, labels, cohps) in enumerate(
                zip(seq_ineq_cations, seq_labels_cohps, seq_cohps)
            ):
                label_str = f"{struct[ication].specie!s}{ication + 1!s}: "
                for label, cohp in zip(labels, cohps):
                    if label is not None:
                        cba_cohp_plot_data[label_str + label] = cohp

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
                summed_spins=False,
                which_bonds=analyse.whichbonds,
                final_dict_bonds=analyse.final_dict_bonds,
                final_dict_ions=analyse.final_dict_ions,
                run_time=cba_run_time,
            )
            if save_cohp_plots:
                describe.plot_cohps(
                    save=True,
                    filename=f"automatic_cohp_plots_{which_bonds}.pdf",
                    hide=True,
                    **plot_kwargs,
                )
                import json

                with open(
                    dir_name / f"condensed_bonding_analysis_{which_bonds}.json", "w"
                ) as fp:
                    json.dump(analyse.condensed_bonding_analysis, fp)
                with open(
                    dir_name / f"condensed_bonding_analysis_{which_bonds}.txt", "w"
                ) as fp:
                    for line in describe.text:
                        fp.write(f"{line}\n")

            # Read in strongest icohp values
            sb_icohp, sb_icobi, sb_icoop = _identify_strongest_bonds(
                analyse=analyse,
                icobilist_path=icobilist_path,
                icohplist_path=icohplist_path,
                icooplist_path=icooplist_path,
            )
            return (
                condensed_bonding_analysis,
                describe,
                sb_icobi,
                sb_icohp,
                sb_icoop,
            )
        except ValueError:
            return (None, None, None, None, None)


class CalcQualitySummary(BaseModel):
    """Model describing the calculation quality of lobster run."""

    minimal_basis: bool = Field(
        None,
        description="Denotes whether the calculation used the minimal basis for the "
        "LOBSTER computation",
    )
    charge_spilling: dict = Field(
        None,
        description="Dict contains the absolute charge spilling value",
    )
    charges: dict = Field(
        None,
        description="Dict contains the LOBSTER and BVA charge sign comparison results",
    )
    band_overlaps: dict = Field(
        None,
        description="Dict summarizing important information from the "
        "bandOverlaps.lobster file to evaluate the quality of the projection, "
        "namely whether the file is generated during projection (i.e., larger "
        "deviations exist), the maximum deviation observed, percent of k-points "
        "above the threshold set in the pymatgen parser (during data generation the "
        "value was set to 0.1)",
    )
    dos_comparisons: dict = Field(
        None,
        description="Dict with Tanimoto index values obtained from comparing "
        "VASP and LOBSTER projected DOS fingerprints",
    )

    @classmethod
    @requires(Analysis, "lobsterpy must be installed to create an CalcQualitySummary.")
    def from_directory(
        cls,
        dir_name: Union[Path, str],
        calc_quality_kwargs: dict = None,
    ):
        dir_name = Path(dir_name)
        band_overlaps_path = dir_name / "bandOverlaps.lobster.gz"
        charge_path = dir_name / "CHARGE.lobster.gz"
        doscar_path = (
            dir_name / "DOSCAR.LSO.lobster.gz"
            if (dir_name / "DOSCAR.LSO.lobster.gz").exists()
            else dir_name / "DOSCAR.lobster.gz"
        )
        lobsterin_path = dir_name / "lobsterin.gz"
        lobsterout_path = dir_name / "lobsterout.gz"
        potcar_path = (
            dir_name / "POTCAR.gz" if (dir_name / "POTCAR.gz").exists() else None
        )
        structure_path = dir_name / "POSCAR.gz"
        vasprun_path = dir_name / "vasprun.xml.gz"

        calc_quality_kwargs = {} if calc_quality_kwargs is None else calc_quality_kwargs
        cal_quality_dict = Analysis.get_lobster_calc_quality_summary(
            path_to_poscar=structure_path,
            path_to_vasprun=vasprun_path,
            path_to_charge=charge_path,
            path_to_potcar=potcar_path,
            path_to_doscar=doscar_path,
            path_to_lobsterin=lobsterin_path,
            path_to_lobsterout=lobsterout_path,
            path_to_bandoverlaps=band_overlaps_path,
            **calc_quality_kwargs,
        )
        return CalcQualitySummary(**cal_quality_dict)


class StrongestBonds(BaseModel):
    """Strongest bonds extracted from ICOHPLIST/ICOOPLIST/ICOBILIST from LOBSTER.

    LobsterPy is used for the extraction.
    """

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
        description="Dict with infos on bond strength and bond length,.",
    )


class LobsterTaskDocument(StructureMetadata):
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
    lobsterpy_text: str = Field(
        None,
        description="Stores LobsterPy automatic analysis summary text",
    )
    calc_quality_summary: CalcQualitySummary = Field(
        None,
        description="Model summarizing results of lobster runs like charge spillings, "
        "band overlaps, DOS comparisons with VASP runs and quantum chemical LOBSTER "
        "charge sign comparisons with BVA method",
    )
    calc_quality_text: str = Field(
        None,
        description="Stores calculation quality analysis summary text",
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
    lobsterpy_data_cation_anion: CondensedBondingAnalysis = Field(
        None, description="Model describing the LobsterPy data"
    )
    lobsterpy_text_cation_anion: str = Field(
        None,
        description="Stores LobsterPy automatic analysis summary text",
    )
    strongest_bonds_icohp_cation_anion: StrongestBonds = Field(
        None, description="Describes the strongest cation-anion ICOHP bonds"
    )
    strongest_bonds_icoop_cation_anion: StrongestBonds = Field(
        None, description="Describes the strongest cation-anion ICOOP bonds"
    )
    strongest_bonds_icobi_cation_anion: StrongestBonds = Field(
        None, description="Describes the strongest cation-anion ICOBI bonds"
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
    site_potentials: dict = Field(
        None,
        description="Site potentials dict from"
        " LOBSTER based on Mulliken and Loewdin charges",
    )
    gross_populations: dict = Field(
        None,
        description="Gross populations dict from"
        " LOBSTER based on Mulliken and Loewdin charges with"
        "each site as a key and the gross population as a value.",
    )
    band_overlaps: dict = Field(
        None,
        description="Band overlaps data for each k-point from"
        " bandOverlaps.lobster file if it exists",
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
        calc_quality_kwargs: dict = None,
        save_cba_jsons: bool = True,
        add_coxxcar_to_task_document: bool = True,
        save_computational_data_jsons: bool = True,
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
        calc_quality_kwargs : dict
            kwargs to change calc quality summary options in lobsterpy.
        save_cba_jsons : bool
            Bool to indicate whether condensed bonding analysis jsons
            should be saved
        add_coxxcar_to_task_document : bool
            Bool to indicate whether to add COHPCAR, COOPCAR, COBICAR data objects
            to the task document
        save_computational_data_jsons : bool
            Bool to indicate whether computational data jsons
            should be saved

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
        charge_path = dir_name / "CHARGE.lobster.gz"
        cobicar_path = dir_name / "COBICAR.lobster.gz"
        coopcar_path = dir_name / "COOPCAR.lobster.gz"
        doscar_path = dir_name / "DOSCAR.lobster.gz"
        structure_path = dir_name / "POSCAR.gz"
        madelung_energies_path = dir_name / "MadelungEnergies.lobster.gz"
        site_potentials_path = dir_name / "SitePotentials.lobster.gz"
        gross_populations_path = dir_name / "GROSSPOP.lobster.gz"
        band_overlaps_path = dir_name / "bandOverlaps.lobster.gz"

        # Do automatic bonding analysis with LobsterPy
        condensed_bonding_analysis = None
        sb_icobi = None
        sb_icohp = None
        sb_icoop = None
        describe = None
        struct = Structure.from_file(structure_path)

        # will perform two condensed bonding analysis computations
        if icohplist_path.exists() and cohpcar_path.exists() and charge_path.exists():
            (
                condensed_bonding_analysis,
                describe,
                sb_icobi,
                sb_icohp,
                sb_icoop,
            ) = CondensedBondingAnalysis.from_directory(
                dir_name,
                save_cohp_plots=save_cohp_plots,
                plot_kwargs=plot_kwargs,
                which_bonds="all",
            )
            (
                condensed_bonding_analysis_ionic,
                describe_ionic,
                sb_icobi_ionic,
                sb_icohp_ionic,
                sb_icoop_ionic,
            ) = CondensedBondingAnalysis.from_directory(
                dir_name,
                save_cohp_plots=save_cohp_plots,
                plot_kwargs=plot_kwargs,
                which_bonds="cation-anion",
            )
        # Get lobster calculation quality summary data
        calc_quality_kwargs_default = {
            "e_range": [-20, 0],
            "dos_comparison": True,
            "n_bins": 256,
            "bva_comp": True,
        }
        if calc_quality_kwargs:
            for args, values in calc_quality_kwargs.items():
                calc_quality_kwargs_default[args] = values

        calc_quality_summary = CalcQualitySummary.from_directory(
            dir_name, calc_quality_kwargs=calc_quality_kwargs_default
        )

        calc_quality_text = Description.get_calc_quality_description(
            calc_quality_summary.dict()
        )

        # Read in charges
        charges = None
        if charge_path.exists():
            charge = Charge(charge_path)
            charges = {"Mulliken": charge.Mulliken, "Loewdin": charge.Loewdin}

        # Read in DOS
        dos = None
        if doscar_path.exists():
            doscar_lobster = Doscar(doscar=doscar_path, structure_file=structure_path)
            dos = doscar_lobster.completedos

        # Read in LSO DOS
        lso_dos = None
        doscar_lso_path = dir_name / "DOSCAR.LSO.lobster.gz"
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

        # Read in Site Potentials
        site_potentials = None
        if site_potentials_path.exists():
            site_potentials_obj = SitePotential(filename=site_potentials_path)

            site_potentials = {
                "Mulliken": site_potentials_obj.sitepotentials_Mulliken,
                "Loewdin": site_potentials_obj.sitepotentials_Loewdin,
                "Ewald_splitting": site_potentials_obj.ewald_splitting,
            }

        # Read in Gross Populations
        gross_populations = None
        if gross_populations_path.exists():
            gross_populations_obj = Grosspop(filename=gross_populations_path)

            gross_populations = {}
            for atom_index, gross_pop in enumerate(
                gross_populations_obj.list_dict_grosspop
            ):
                gross_populations[atom_index] = gross_pop

        # Read in Band overlaps
        band_overlaps = None
        if band_overlaps_path.exists():
            band_overlaps_obj = Bandoverlaps(filename=band_overlaps_path)

            band_overlaps = {}
            for spin, value in band_overlaps_obj.bandoverlapsdict.items():
                band_overlaps[str(spin.value)] = value

        # Read in COHPCAR, COBICAR, COOPCAR
        cohp_obj = None
        coop_obj = None
        cobi_obj = None

        if add_coxxcar_to_task_document:
            if cohpcar_path.exists():
                cohp_obj = CompleteCohp.from_file(
                    fmt="LOBSTER",
                    structure_file=structure_path,
                    filename=cohpcar_path,
                    are_coops=False,
                    are_cobis=False,
                )

            if coopcar_path.exists():
                coop_obj = CompleteCohp.from_file(
                    fmt="LOBSTER",
                    structure_file=structure_path,
                    filename=coopcar_path,
                    are_coops=True,
                    are_cobis=False,
                )

            if cobicar_path.exists():
                cobi_obj = CompleteCohp.from_file(
                    fmt="LOBSTER",
                    structure_file=structure_path,
                    filename=cobicar_path,
                    are_coops=False,
                    are_cobis=True,
                )

        doc = cls.from_structure(
            structure=struct,
            meta_structure=struct,
            dir_name=dir_name,
            lobsterin=lobster_in,
            lobsterout=lobster_out,
            # include additional fields for cation-anion
            lobsterpy_data=condensed_bonding_analysis,
            lobsterpy_text=" ".join(describe.text) if describe is not None else None,
            strongest_bonds_icohp=sb_icohp,
            strongest_bonds_icoop=sb_icoop,
            strongest_bonds_icobi=sb_icobi,
            lobsterpy_data_cation_anion=condensed_bonding_analysis_ionic,
            lobsterpy_text_cation_anion=" ".join(describe_ionic.text)
            if describe_ionic is not None
            else None,
            strongest_bonds_icohp_cation_anion=sb_icohp_ionic,
            strongest_bonds_icoop_cation_anion=sb_icoop_ionic,
            strongest_bonds_icobi_cation_anion=sb_icobi_ionic,
            calc_quality_summary=calc_quality_summary,
            calc_quality_text=" ".join(calc_quality_text),
            dos=dos,
            lso_dos=lso_dos,
            charges=charges,
            madelung_energies=madelung_energies,
            site_potentials=site_potentials,
            gross_populations=gross_populations,
            band_overlaps=band_overlaps,
            # include additional fields for all bonds
            cohp_data=cohp_obj,
            coop_data=coop_obj,
            cobi_data=cobi_obj,
        )

        if save_cba_jsons:
            cba_json_save_dir = dir_name / "cba.json.gz"
            with gzip.open(cba_json_save_dir, "wt", encoding="UTF-8") as f:
                f.write("[")
                if (
                    doc.lobsterpy_data_cation_anion is not None
                ):  # check if cation-anion analysis failed
                    lobsterpy_analysis_type = (
                        doc.lobsterpy_data_cation_anion.which_bonds.replace("-", "_")
                    )
                    cation_anion_bonds_dict = {
                        f"{lobsterpy_analysis_type}_bonds": {
                            "lobsterpy_data": doc.lobsterpy_data_cation_anion.dict(),
                            "lobsterpy_text": [
                                "".join(doc.lobsterpy_text_cation_anion)
                            ],
                            "sb_icobi": doc.strongest_bonds_icobi_cation_anion.dict(),
                            "sb_icohp": doc.strongest_bonds_icohp_cation_anion.dict(),
                            "sb_icoop": doc.strongest_bonds_icoop_cation_anion.dict(),
                        }
                    }

                    for item in cation_anion_bonds_dict[
                        f"{lobsterpy_analysis_type}_bonds"
                    ]["lobsterpy_data"]["cohp_plot_data"].items():
                        plot_label, cohps = item
                        # check if item has a `as_dict` method
                        # (i.e. it is a pymatgen object)
                        if hasattr(cohps, "as_dict"):
                            cation_anion_bonds_dict[f"{lobsterpy_analysis_type}_bonds"][
                                "lobsterpy_data"
                            ]["cohp_plot_data"][plot_label] = cohps.as_dict()
                    _replace_inf_values(cation_anion_bonds_dict)
                    json.dump(cation_anion_bonds_dict, f)
                    f.write(",")  # add comma separator between two dicts
                else:
                    lobsterpy_analysis_type = "cation_anion"
                    cation_anion_bonds_dict = {f"{lobsterpy_analysis_type}_bonds": {}}
                    json.dump(cation_anion_bonds_dict, f)
                    f.write(",")  # add comma separator between two dicts

                # add all-bonds data
                lobsterpy_analysis_type = doc.lobsterpy_data.which_bonds
                all_bonds_data = {
                    f"{lobsterpy_analysis_type}_bonds": {
                        "lobsterpy_data": doc.lobsterpy_data.dict(),
                        "lobsterpy_text": ["".join(doc.lobsterpy_text)],
                        "sb_icobi": doc.strongest_bonds_icobi.dict(),
                        "sb_icohp": doc.strongest_bonds_icohp.dict(),
                        "sb_icoop": doc.strongest_bonds_icoop.dict(),
                    }
                }

                for item in all_bonds_data[f"{lobsterpy_analysis_type}_bonds"][
                    "lobsterpy_data"
                ]["cohp_plot_data"].items():
                    plot_label, cohps = item
                    # check if item has a `as_dict` method
                    # (i.e. it is a pymatgen object)
                    if hasattr(cohps, "as_dict"):
                        all_bonds_data[f"{lobsterpy_analysis_type}_bonds"][
                            "lobsterpy_data"
                        ]["cohp_plot_data"][plot_label] = cohps.as_dict()
                _replace_inf_values(all_bonds_data)
                json.dump(all_bonds_data, f)
                f.write(",")  # add comma separator between two dicts
                json.dump(
                    {"madelung_energies": doc.madelung_energies}, f
                )  # add madelung energies
                f.write(",")
                json.dump({"charges": doc.charges}, f)  # add charges
                f.write(",")
                json.dump(
                    {"calc_quality_summary": doc.calc_quality_summary.dict()}, f
                )  # add calc quality summary dict
                f.write(",")
                json.dump(
                    {"calc_quality_text": ["".join(doc.calc_quality_text)]}, f
                )  # add calc quality text
                f.write(",")
                json.dump({"dos": doc.dos.as_dict()}, f)  # add dos of lobster
                f.write(",")
                json.dump(
                    {
                        "lso_dos": doc.lso_dos.as_dict()
                        if doc.lso_dos is not None
                        else {}
                    },
                    f,
                )  # add los dos of lobster if exist
                f.write(",")
                builder_meta = doc.builder_meta.dict()
                builder_meta["build_date"] = str(doc.builder_meta.build_date.utcnow())
                json.dump(
                    {"builder_meta": builder_meta},
                    f,
                )  # add builder metadata
                f.write("]")

        if save_computational_data_jsons:
            computational_data_json_save_dir = dir_name / "computational_data.json.gz"
            fields_to_exclude = [
                "nsites",
                "elements",
                "nelements",
                "composition",
                "composition_reduced",
                "formula_pretty",
                "formula_anonymous",
                "chemsys",
                "volume",
                "density",
                "density_atomic",
                "symmetry",
                "dir_name",
                "last_updated",
            ]
            doc_here = doc.copy(deep=True, update=additional_fields)

            if cohpcar_path.exists() and doc.cohp_data is None:
                cohp_obj = CompleteCohp.from_file(
                    fmt="LOBSTER",
                    structure_file=structure_path,
                    filename=cohpcar_path,
                    are_coops=False,
                    are_cobis=False,
                )
                doc_here.__setattr__("cohp_data", cohp_obj)

            if coopcar_path.exists() and doc.coop_data is None:
                coop_obj = CompleteCohp.from_file(
                    fmt="LOBSTER",
                    structure_file=structure_path,
                    filename=coopcar_path,
                    are_coops=True,
                    are_cobis=False,
                )
                doc_here.__setattr__("coop_data", coop_obj)

            if cobicar_path.exists() and doc.cobi_data is None:
                cobi_obj = CompleteCohp.from_file(
                    fmt="LOBSTER",
                    structure_file=structure_path,
                    filename=cobicar_path,
                    are_coops=False,
                    are_cobis=True,
                )
                doc_here.__setattr__("cobi_data", cobi_obj)
            with gzip.open(
                computational_data_json_save_dir, "wt", encoding="UTF-8"
            ) as f:
                f.write("[")
                for attribute in doc_here.__fields__:
                    if attribute not in fields_to_exclude:
                        if hasattr(doc_here.__getattribute__(attribute), "dict"):
                            if "lobsterpy_data" in attribute:
                                data = doc_here.__getattribute__(attribute).dict()
                                for item in data["cohp_plot_data"].items():
                                    key, value = item
                                    if hasattr(
                                        value, "as_dict"
                                    ):  # check if item has a `as_dict` method
                                        # (i.e. it is a pymatgen object)
                                        data["cohp_plot_data"][key] = value.as_dict()
                                data_new = {attribute: data}
                                _replace_inf_values(data_new)
                                json.dump(data_new, f)
                                if (
                                    attribute != list(doc_here.__fields__.keys())[-1]
                                ):  # add comma separator between two dicts
                                    f.write(",")
                            else:
                                if attribute == "builder_meta":
                                    builder_meta = doc_here.__getattribute__(
                                        attribute
                                    ).dict()
                                    builder_meta["build_date"] = str(
                                        doc.builder_meta.build_date.utcnow()
                                    )
                                    data = {attribute: builder_meta}
                                else:
                                    data = {
                                        attribute: doc_here.__getattribute__(
                                            attribute
                                        ).dict()
                                    }
                                json.dump(data, f)
                                if (
                                    attribute != list(doc_here.__fields__.keys())[-1]
                                ):  # add comma separator between two dicts
                                    f.write(",")
                        elif hasattr(doc_here.__getattribute__(attribute), "as_dict"):
                            data = {
                                attribute: doc_here.__getattribute__(
                                    attribute
                                ).as_dict()
                            }
                            json.dump(data, f)
                            if (
                                attribute != list(doc_here.__fields__.keys())[-1]
                            ):  # add comma separator between two dicts
                                f.write(",")
                        else:
                            data = {attribute: doc_here.__getattribute__(attribute)}
                            json.dump(data, f)
                            if (
                                attribute != list(doc_here.__fields__.keys())[-1]
                            ):  # add comma separator between two dicts
                                f.write(",")
                f.write("]")

        return doc.copy(update=additional_fields)


def _replace_inf_values(data: Union[dict[Any, Any], list[Any]]):
    """
    Replace the -inf value in dictionary and with the string representation '-Infinity'.

    Parameters
    ----------
    data : dict
        dictionary to recursively iterate over

    Returns
    -------
    data
        Dictionary with replaced -inf values.

    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                _replace_inf_values(
                    value
                )  # Recursively process nested dictionaries and lists
            elif value == float("-inf"):
                data[key] = "-Infinity"  # Replace -inf with a string representation
    elif isinstance(data, list):
        for index, item in enumerate(data):
            if isinstance(item, (dict, list)):
                _replace_inf_values(
                    item
                )  # Recursively process nested dictionaries and lists
            elif item == float("-inf"):
                data[index] = "-Infinity"  # Replace -inf with a string representation


def _identify_strongest_bonds(
    analyse: Analysis,
    icobilist_path: Path,
    icohplist_path: Path,
    icooplist_path: Path,
):
    """
    Identify the strongest bonds and convert them into StrongestBonds objects.

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
                filename=file,
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


# Don't we have this in pymatgen somewhere?
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
    for a, b, c, length in zip(
        bondlist["list_atom1"],
        bondlist["list_atom2"],
        bondlist["list_icohp"],
        bondlist["list_length"],
    ):
        bonds.append(f"{a.rstrip('0123456789')}-{b.rstrip('0123456789')}")
        icohp_all.append(sum(c.values()))
        lengths.append(length)

    bond_labels_unique = list(set(bonds))
    sep_icohp: List[List[float]] = [[] for _ in range(len(bond_labels_unique))]
    sep_lengths: List[List[float]] = [[] for _ in range(len(bond_labels_unique))]

    for i, val in enumerate(bond_labels_unique):
        for j, val2 in enumerate(bonds):
            if val == val2:
                sep_icohp[i].append(icohp_all[j])
                sep_lengths[i].append(lengths[j])

    if are_cobis and not are_coops:
        prop = "ICOBI"
    elif not are_cobis and are_coops:
        prop = "ICOOP"
    else:
        prop = "ICOHP"

    bond_dict = {}
    for i, lab in enumerate(bond_labels_unique):
        label = lab.split("-")
        label.sort()
        for rel_bnd in relevant_bonds:
            rel_bnd_list = rel_bnd.split("-")
            rel_bnd_list.sort()
            if label == rel_bnd_list:
                if prop == "ICOHP":
                    index = np.argmin(sep_icohp[i])
                    bond_dict.update(
                        {
                            rel_bnd: {
                                prop: min(sep_icohp[i]),
                                "length": sep_lengths[i][index],
                            }
                        }
                    )
                else:
                    index = np.argmax(sep_icohp[i])
                    bond_dict.update(
                        {
                            rel_bnd: {
                                prop: max(sep_icohp[i]),
                                "length": sep_lengths[i][index],
                            }
                        }
                    )
    return bond_dict


def read_saved_json(
    filename: str, pymatgen_objs: bool = True, query: str = "structure"
):
    """
    Read the data from  *.json.gz file corresponding to query.

    Parameters
    ----------
    filename: str
        name of the json file to read
    pymatgen_objs: bool
        if True will convert structure,coop, cobi, cohp and dos to pymatgen objects
    query: str
        field name to query from the json file.

    Returns
    -------
    dict
        Returns a dictionary with lobster task json data corresponding to query.
    """
    lobster_data = {}
    with gzip.open(filename, "rb") as f:
        objects = ijson.items(f, "item", use_float=True)
        for obj in objects:
            if query in obj:
                for field, data in obj.items():
                    if pymatgen_objs:
                        if field == "structure":
                            lobster_data[field] = Structure.from_dict(data)
                        elif (
                            field == "lobsterpy_data"
                            or field == "lobsterpy_data_cation_anion"
                        ):
                            lobster_data[field] = data
                            if lobster_data[field]:
                                for plotlabel, cohp in lobster_data[field][
                                    "cohp_plot_data"
                                ].items():
                                    lobster_data[field]["cohp_plot_data"][
                                        plotlabel
                                    ] = Cohp.from_dict(cohp)
                        elif field == "all_bonds" or field == "cation_anion_bonds":
                            lobster_data[field] = data
                            if lobster_data[field]:
                                for plotlabel, cohp in lobster_data[field][
                                    "lobsterpy_data"
                                ]["cohp_plot_data"].items():
                                    lobster_data[field]["lobsterpy_data"][
                                        "cohp_plot_data"
                                    ][plotlabel] = Cohp.from_dict(cohp)
                        elif (
                            field == "cohp_data"
                            or field == "cobi_data"
                            or field == "coop_data"
                        ):
                            if data:
                                lobster_data[field] = CompleteCohp.from_dict(data)
                            else:
                                lobster_data[field] = None
                        elif field == "lso_dos" or field == "dos":
                            if data:
                                lobster_data[field] = LobsterCompleteDos.from_dict(data)
                            else:
                                lobster_data[field] = None
                        else:
                            lobster_data[field] = data
                    else:
                        lobster_data[field] = data
                break

    return lobster_data
