"""Module defining lobster document schemas."""

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Union

import ijson
import numpy as np
from emmet.core.structure import StructureMetadata
from monty.dev import requires
from monty.json import MontyDecoder, jsanitize
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.electronic_structure.cohp import CompleteCohp
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

from atomate2 import SETTINGS, __version__
from atomate2.lobster.files import FileNames
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

    gaussiansmearingwidth: Optional[float] = Field(
        None, description="Set the smearing width in eV,default is 0.2 (eV)"
    )
    usedecimalplaces: Optional[int] = Field(
        None,
        description="Set the decimal places to print in output files, default is 5",
    )
    cohpsteps: Optional[float] = Field(
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
    lsodos: Optional[bool] = Field(
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
        lobster_zip_files: bool = SETTINGS.LOBSTER_ZIP_FILES,  # type:ignore[assignment]
    ) -> tuple:
        """
        Create a task document from a directory containing LOBSTER files.

        Parameters
        ----------
        dir_name : path or str
            The path to the folder containing the calculation outputs.
        save_cohp_plots : bool.
            Bool to indicate whether automatic cohp plots and jsons
            from lobsterpy will be generated.
        plot_kwargs : dict.
            kwargs to change plotting options in lobsterpy.
        which_bonds: str.
            mode for condensed bonding analysis: "cation-anion" and "all".
        lobster_zip_files: bool.
            boolean indicating if the files in the directory are gzipped
        """
        plot_kwargs = plot_kwargs or {}
        dir_name = Path(dir_name)
        file_names = FileNames(lobster_zip_files=lobster_zip_files)
        cohpcar_path = (
            dir_name / file_names.cohpcar_lobster  # type:ignore[attr-defined]
        )
        charge_path = dir_name / file_names.charge_lobster  # type:ignore[attr-defined]
        structure_path = dir_name / file_names.poscar  # type:ignore[attr-defined]
        icohplist_path = (
            dir_name / file_names.icohplist_lobster  # type:ignore[attr-defined]
        )
        icobilist_path = (
            dir_name / file_names.icobilist_lobster  # type:ignore[attr-defined]
        )
        icooplist_path = (
            dir_name / file_names.icooplist_lobster  # type:ignore[attr-defined]
        )

        try:
            start = time.time()
            analyse = Analysis(
                path_to_poscar=structure_path,
                path_to_icohplist=icohplist_path,
                path_to_cohpcar=cohpcar_path,
                path_to_charge=charge_path,
                summed_spins=False,  # we will always use spin polarization here
                cutoff_icohp=0.10,
                which_bonds=which_bonds,
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
                which_bonds=analyse.which_bonds,
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
            return None, None, None, None, None


class DosComparisons(BaseModel):
    """Model describing the DOS comparisons field in the CalcQualitySummary model."""

    tanimoto_orb_s: Optional[float] = Field(
        None,
        description="Tanimoto similarity index between s orbital of "
        "VASP and LOBSTER DOS",
    )
    tanimoto_orb_p: Optional[float] = Field(
        None,
        description="Tanimoto similarity index between p orbital of "
        "VASP and LOBSTER DOS",
    )
    tanimoto_orb_d: Optional[float] = Field(
        None,
        description="Tanimoto similarity index between d orbital of "
        "VASP and LOBSTER DOS",
    )
    tanimoto_orb_f: Optional[float] = Field(
        None,
        description="Tanimoto similarity index between f orbital of "
        "VASP and LOBSTER DOS",
    )
    tanimoto_summed: float = Field(
        None,
        description="Tanimoto similarity index for summed PDOS between "
        "VASP and LOBSTER",
    )
    e_range: list[float] = Field(
        None,
        description="Energy range used for evaluating the Tanimoto similarity index",
    )
    n_bins: int = Field(
        None,
        description="Number of bins used for discretizing the VASP and LOBSTER PDOS"
        "(Affects the Tanimoto similarity index)",
    )


class ChargeComparisons(BaseModel):
    """Model describing the charges field in the CalcQualitySummary model."""

    BVA_Mulliken_agree: Optional[bool] = Field(
        None,
        description="Bool indicating whether atoms classification as cation "
        "or anion based on Mulliken charge signs of LOBSTER "
        "agree with BVA analysis",
    )
    BVA_Loewdin_agree: Optional[bool] = Field(
        None,
        description="Bool indicating whether atoms classification as cations "
        "or anions based on Loewdin charge signs of LOBSTER "
        "agree with BVA analysis",
    )


class BandOverlapsComparisons(BaseModel):
    """Model describing the Band overlaps field in the CalcQualitySummary model."""

    file_exists: bool = Field(
        None,
        description="Boolean indicating whether the bandOverlaps.lobster "
        "file is generated during the LOBSTER run",
    )
    limit_maxDeviation: Optional[float] = Field(
        None,
        description="Limit set for maximal deviation in pymatgen parser",
    )
    has_good_quality_maxDeviation: bool = Field(
        None,
        description="Boolean indicating whether the deviation at each k-point "
        "is within the threshold set using limit_maxDeviation "
        "for analyzing the bandOverlaps.lobster file data",
    )
    max_deviation: Optional[float] = Field(
        None,
        description="Maximum deviation from ideal identity matrix from the observed in "
        "the bandOverlaps.lobster file",
    )
    percent_kpoints_abv_limit: Optional[float] = Field(
        None,
        description="Percent of k-points that show deviations above "
        "the limit_maxDeviation threshold set in pymatgen parser.",
    )


class ChargeSpilling(BaseModel):
    """Model describing the Charge spilling field in the CalcQualitySummary model."""

    abs_charge_spilling: float = Field(
        None,
        description="Absolute charge spilling value from the LOBSTER calculation.",
    )
    abs_total_spilling: float = Field(
        None,
        description="Total charge spilling percent from the LOBSTER calculation.",
    )


class CalcQualitySummary(BaseModel):
    """Model describing the calculation quality of lobster run."""

    minimal_basis: bool = Field(
        None,
        description="Denotes whether the calculation used the minimal basis for the "
        "LOBSTER computation",
    )
    charge_spilling: ChargeSpilling = Field(
        None,
        description="Model describing the charge spilling from the LOBSTER runs",
    )
    charges: ChargeComparisons = Field(
        None,
        description="Model describing the charge sign comparison results",
    )
    band_overlaps: BandOverlapsComparisons = Field(
        None,
        description="Model describing the band overlap file analysis results",
    )
    dos_comparisons: DosComparisons = Field(
        None,
        description="Model describing the VASP and LOBSTER PDOS comparisons results",
    )

    @classmethod
    @requires(Analysis, "lobsterpy must be installed to create an CalcQualitySummary.")
    def from_directory(
        cls,
        dir_name: Union[Path, str],
        calc_quality_kwargs: dict = None,
        lobster_zip_files: bool = SETTINGS.LOBSTER_ZIP_FILES,  # type:ignore[assignment]
    ):
        """
        Create a LOBSTER calculation quality summary from directory with LOBSTER files.

        Parameters
        ----------
        dir_name : path or str
            The path to the folder containing the calculation outputs.
        calc_quality_kwargs : dict
            kwargs to change calc quality analysis options in lobsterpy.
        lobster_zip_files: bool.
            boolean indicating if the files in the directory are gzipped

        Returns
        -------
        CalcQualitySummary
            A task document summarizing quality of the lobster calculation.
        """
        dir_name = Path(dir_name)
        file_names = FileNames(
            lobster_zip_files=lobster_zip_files  # type:ignore
        )
        band_overlaps_path = (
            dir_name / file_names.bandoverlaps_lobster  # type:ignore[attr-defined]
        )
        charge_path = dir_name / file_names.charge_lobster  # type:ignore[attr-defined]
        doscar_path = (
            dir_name / file_names.doscar_lso_lobster  # type:ignore[attr-defined]
            if (
                dir_name / file_names.doscar_lso_lobster  # type:ignore[attr-defined]
            ).exists()
            else dir_name / file_names.doscar_lobster  # type:ignore[attr-defined]
        )
        lobsterin_path = dir_name / file_names.lobsterin  # type:ignore[attr-defined]
        lobsterout_path = dir_name / file_names.lobsterout  # type:ignore[attr-defined]
        potcar_path = (
            dir_name / file_names.potcar  # type:ignore[attr-defined]
            if (dir_name / file_names.potcar).exists()  # type:ignore[attr-defined]
            else None
        )
        structure_path = dir_name / file_names.poscar  # type:ignore[attr-defined]
        vasprun_path = dir_name / file_names.vasprun_xml  # type:ignore[attr-defined]

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
    lobsterpy_data_cation_anion: Optional[CondensedBondingAnalysis] = Field(
        None, description="Model describing the LobsterPy data"
    )
    lobsterpy_text_cation_anion: Optional[str] = Field(
        None,
        description="Stores LobsterPy automatic analysis summary text",
    )
    strongest_bonds_icohp_cation_anion: Optional[StrongestBonds] = Field(
        None, description="Describes the strongest cation-anion ICOHP bonds"
    )
    strongest_bonds_icoop_cation_anion: Optional[StrongestBonds] = Field(
        None, description="Describes the strongest cation-anion ICOOP bonds"
    )
    strongest_bonds_icobi_cation_anion: Optional[StrongestBonds] = Field(
        None, description="Describes the strongest cation-anion ICOBI bonds"
    )
    dos: LobsterCompleteDos = Field(
        None, description="pymatgen pymatgen.io.lobster.Doscar.completedos data"
    )
    lso_dos: Optional[LobsterCompleteDos] = Field(
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
    band_overlaps: Optional[dict] = Field(
        None,
        description="Band overlaps data for each k-point from"
        " bandOverlaps.lobster file if it exists",
    )
    cohp_data: Optional[CompleteCohp] = Field(
        None, description="pymatgen CompleteCohp object with COHP data"
    )
    coop_data: Optional[CompleteCohp] = Field(
        None, description="pymatgen CompleteCohp object with COOP data"
    )
    cobi_data: Optional[CompleteCohp] = Field(
        None, description="pymatgen CompleteCohp object with COBI data"
    )

    schema: str = Field(
        __version__, description="Version of atomate2 used to create the document"
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
        lobster_zip_files: bool = SETTINGS.LOBSTER_ZIP_FILES,  # type:ignore[assignment]
    ) -> "LobsterTaskDocument":
        """
        Create a task document from a directory containing LOBSTER files.

        Parameters
        ----------
        dir_name : path or str.
            The path to the folder containing the calculation outputs.
        additional_fields : dict.
            Dictionary of additional fields to add to output document.
        store_lso_dos : bool.
            Whether to store the LSO DOS.
        save_cohp_plots : bool.
            Bool to indicate whether automatic cohp plots and jsons
            from lobsterpy will be generated.
        plot_kwargs : dict.
            kwargs to change plotting options in lobsterpy.
        calc_quality_kwargs : dict.
            kwargs to change calc quality summary options in lobsterpy.
        save_cba_jsons : bool.
            Bool to indicate whether condensed bonding analysis jsons
            should be saved, consists of outputs from lobsterpy analysis,
            calculation quality summary, lobster dos, charges and madelung energies
        add_coxxcar_to_task_document : bool.
            Bool to indicate whether to add COHPCAR, COOPCAR, COBICAR data objects
            to the task document
        save_computational_data_jsons : bool.
            Bool to indicate whether computational data jsons
            should be saved
        lobster_zip_files: bool.
            boolean indicating if the files in the directory are gzipped

        Returns
        -------
        LobsterTaskDocument
            A task document for the lobster calculation.
        """
        additional_fields = additional_fields or {}
        dir_name = Path(dir_name)

        file_names = FileNames(
            lobster_zip_files=lobster_zip_files  # type: ignore[assignment]
        )

        # Read in lobsterout and lobsterin
        lobsterout_doc = Lobsterout(
            dir_name / file_names.lobsterout  # type:ignore[attr-defined]
        ).get_doc()
        lobster_out = LobsteroutModel(**lobsterout_doc)
        lobster_in = LobsterinModel(
            **Lobsterin.from_file(
                dir_name / file_names.lobsterin  # type:ignore[attr-defined]
            )
        )

        icohplist_path = (
            dir_name / file_names.icohplist_lobster  # type:ignore[attr-defined]
        )
        cohpcar_path = (
            dir_name / file_names.cohpcar_lobster  # type:ignore[attr-defined]
        )
        charge_path = dir_name / file_names.charge_lobster  # type:ignore[attr-defined]
        cobicar_path = (
            dir_name / file_names.cobicar_lobster  # type:ignore[attr-defined]
        )
        coopcar_path = (
            dir_name / file_names.coopcar_lobster  # type:ignore[attr-defined]
        )
        doscar_path = dir_name / file_names.doscar_lobster  # type:ignore[attr-defined]
        structure_path = dir_name / file_names.poscar  # type:ignore[attr-defined]
        madelung_energies_path = (
            dir_name / file_names.madelungenergies_lobster  # type:ignore[attr-defined]
        )
        site_potentials_path = (
            dir_name / file_names.sitepotentials_lobster  # type:ignore[attr-defined]
        )
        gross_populations_path = (
            dir_name / file_names.grosspop_lobster  # type:ignore[attr-defined]
        )
        band_overlaps_path = (
            dir_name / file_names.bandoverlaps_lobster  # type:ignore[attr-defined]
        )

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
                lobster_zip_files=lobster_zip_files,
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
                lobster_zip_files=lobster_zip_files,
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
            dir_name,
            calc_quality_kwargs=calc_quality_kwargs_default,
            lobster_zip_files=lobster_zip_files,
        )

        calc_quality_text = Description.get_calc_quality_description(
            calc_quality_summary.model_dump()
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
        doscar_lso_path = (
            dir_name / file_names.doscar_lso_lobster  # type:ignore[attr-defined]
        )
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

        _replace_inf_values(doc.lobsterpy_data.limit_icohp)
        if doc.lobsterpy_data_cation_anion is not None:
            _replace_inf_values(doc.lobsterpy_data_cation_anion.limit_icohp)

        if save_cba_jsons:
            cba_json_save_dir = dir_name / "cba.json.gz"
            with gzip.open(cba_json_save_dir, "wt", encoding="UTF-8") as f:
                # Write the json in iterable format
                # (Necessary to load large JSON files via ijson)
                f.write("[")
                if (
                    doc.lobsterpy_data_cation_anion is not None
                ):  # check if cation-anion analysis failed
                    lobsterpy_analysis_type = (
                        doc.lobsterpy_data_cation_anion.which_bonds.replace("-", "_")
                    )
                    data = {
                        f"{lobsterpy_analysis_type}_bonds": {
                            "lobsterpy_data": doc.lobsterpy_data_cation_anion,
                            "lobsterpy_text": [
                                "".join(doc.lobsterpy_text_cation_anion)
                            ],
                            "sb_icobi": doc.strongest_bonds_icobi_cation_anion,
                            "sb_icohp": doc.strongest_bonds_icohp_cation_anion,
                            "sb_icoop": doc.strongest_bonds_icoop_cation_anion,
                        }
                    }
                else:
                    data = {"cation_anion_bonds": {}}
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, f)
                f.write(",")
                # add all-bonds data
                lobsterpy_analysis_type = doc.lobsterpy_data.which_bonds
                data = {
                    f"{lobsterpy_analysis_type}_bonds": {
                        "lobsterpy_data": doc.lobsterpy_data,
                        "lobsterpy_text": ["".join(doc.lobsterpy_text)],
                        "sb_icobi": doc.strongest_bonds_icobi,
                        "sb_icohp": doc.strongest_bonds_icohp,
                        "sb_icoop": doc.strongest_bonds_icoop,
                    }
                }
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, f)
                f.write(",")
                data = {
                    "madelung_energies": doc.madelung_energies
                }  # add madelung energies
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, f)
                f.write(",")
                data = {"charges": doc.charges}  # add charges
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, f)
                f.write(",")
                data = {
                    "calc_quality_summary": doc.calc_quality_summary
                }  # add calc quality summary dict
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, f)
                f.write(",")
                data = {
                    "calc_quality_text": [
                        "".join(doc.calc_quality_text)
                    ]  # type: ignore
                }  # add calc quality summary dict
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, f)
                f.write(",")
                data = {"dos": doc.dos}  # add NON LSO of lobster
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, f)
                f.write(",")
                data = {"lso_dos": doc.lso_dos}  # add LSO DOS of lobster
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, f)
                f.write(",")
                data = {"builder_meta": doc.builder_meta}  # add builder metadata
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=False, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, f)
                del data, monty_encoded_json_doc
                f.write("]")

        if save_computational_data_jsons:
            computational_data_json_save_dir = dir_name / "computational_data.json.gz"
            fields_to_exclude = [
                "nsites",
                "elements",
                "nelements",
                "formula_anonymous",
                "chemsys",
                "volume",
                "density",
                "density_atomic",
                "symmetry",
            ]
            # Always add cohp, cobi and coop data to the jsons if files exists
            if cohpcar_path.exists() and doc.cohp_data is None:
                cohp_obj = CompleteCohp.from_file(
                    fmt="LOBSTER",
                    structure_file=structure_path,
                    filename=cohpcar_path,
                    are_coops=False,
                    are_cobis=False,
                )
                doc.__setattr__("cohp_data", cohp_obj)

            if coopcar_path.exists() and doc.coop_data is None:
                coop_obj = CompleteCohp.from_file(
                    fmt="LOBSTER",
                    structure_file=structure_path,
                    filename=coopcar_path,
                    are_coops=True,
                    are_cobis=False,
                )
                doc.__setattr__("coop_data", coop_obj)

            if cobicar_path.exists() and doc.cobi_data is None:
                cobi_obj = CompleteCohp.from_file(
                    fmt="LOBSTER",
                    structure_file=structure_path,
                    filename=cobicar_path,
                    are_coops=False,
                    are_cobis=True,
                )
                doc.__setattr__("cobi_data", cobi_obj)
            with gzip.open(
                computational_data_json_save_dir, "wt", encoding="UTF-8"
            ) as f:
                # Write the json in iterable format
                # (Necessary to load large JSON files via ijson)
                f.write("[")
                for attribute in doc.model_fields:
                    if attribute not in fields_to_exclude:
                        # Use monty encoder to automatically convert pymatgen
                        # objects and other data json compatible dict format
                        data = {
                            attribute: jsanitize(
                                doc.__getattribute__(attribute),
                                allow_bson=False,
                                strict=True,
                                enum_values=True,
                            )
                        }
                        json.dump(data, f)
                        if attribute != list(doc.model_fields.keys())[-1]:
                            f.write(",")  # add comma separator between two dicts
                        del data
                f.write("]")

            # Again unset the cohp, cobi and coop data fields if not desired in the DB
            if not add_coxxcar_to_task_document:
                doc.__setattr__("cohp_data", None)
                doc.__setattr__("coop_data", None)
                doc.__setattr__("cobi_data", None)

        return doc.model_copy(update=additional_fields)


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
) -> list[StrongestBonds]:
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
    list[StrongestBonds]
        List of StrongestBonds
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
                    which_bonds=analyse.which_bonds,
                )
            )
        else:
            output.append(None)
    return output


# Don't we have this in pymatgen somewhere?
def _get_strong_bonds(
    bondlist: dict, are_cobis: bool, are_coops: bool, relevant_bonds: dict
) -> dict:
    """
    Identify the strongest bonds from a list of bonds.

    Parameters
    ----------
    bondlist : dict.
        dict including bonding information
    are_cobis : bool.
        True if these are cobis
    are_coops : bool.
        True if these are coops
    relevant_bonds : dict.
        Dict include all bonds that are considered.

    Returns
    -------
    dict
        Dictionary including the strongest bonds.
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
    sep_icohp: list[list[float]] = [[] for _ in range(len(bond_labels_unique))]
    sep_lengths: list[list[float]] = [[] for _ in range(len(bond_labels_unique))]

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

    Uses ijson to parse specific keys(memory efficient)

    Parameters
    ----------
    filename: str.
        name of the json file to read
    pymatgen_objs: bool.
        if True will convert structure,coop, cobi, cohp and dos to pymatgen objects
    query: str or None.
        field name to query from the json file. If None, all data will be returned.

    Returns
    -------
    dict
        Returns a dictionary with lobster task json data corresponding to query.
    """
    with gzip.open(filename, "rb") as f:
        lobster_data = {}
        objects = ijson.items(f, "item", use_float=True)
        for obj in objects:
            if query is None:
                for field, data in obj.items():
                    lobster_data[field] = data
            elif query in obj:
                for field, data in obj.items():
                    lobster_data[field] = data
                break
        if not lobster_data:
            raise ValueError(
                "Please recheck the query argument. "
                f"No data associated to the requested 'query={query}' "
                f"found in the JSON file"
            )
    if pymatgen_objs:
        for query_key, value in lobster_data.items():
            if isinstance(value, dict):
                lobster_data[query_key] = MontyDecoder().process_decoded(value)
                # Ensure nested pymatgen objects
                # are converted (applicable for cba.json.gz)
                if "lobsterpy_data" in value:
                    for field in lobster_data[query_key]["lobsterpy_data"].__fields__:
                        lobster_data[query_key]["lobsterpy_data"].__setattr__(
                            field,
                            MontyDecoder().process_decoded(
                                lobster_data[query_key][
                                    "lobsterpy_data"
                                ].__getattribute__(field)
                            ),
                        )
                # This ensures nested pymatgen objects are
                # converted (applicable for computational_data.json.gz)
                elif "lobsterpy_data" in query_key:
                    for field in lobster_data[query_key].__fields__:
                        lobster_data[query_key].__setattr__(
                            field,
                            MontyDecoder().process_decoded(
                                lobster_data[query_key].__getattribute__(field)
                            ),
                        )

    return lobster_data
