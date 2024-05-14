"""Module defining lobster document schemas."""

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from emmet.core.structure import StructureMetadata
from monty.dev import requires
from monty.json import MontyDecoder, jsanitize
from monty.os.path import zpath
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
from typing_extensions import Self

from atomate2 import __version__
from atomate2.utils.datetime import datetime_str

try:
    import ijson
    from lobsterpy.cohp.analyze import Analysis
    from lobsterpy.cohp.describe import Description
except ImportError:
    ijson = None
    Analysis = None
    Description = None


logger = logging.getLogger(__name__)


class LobsteroutModel(BaseModel):
    """Definition of computational settings from the LOBSTER computation."""

    restart_from_projection: Optional[bool] = Field(
        None,
        description="Bool indicating if the run has been restarted from a projection",
    )
    lobster_version: Optional[str] = Field(None, description="Lobster version")
    threads: Optional[int] = Field(
        None, description="Number of threads that Lobster ran on"
    )
    dft_program: Optional[str] = Field(
        None, description="DFT program was used for this run"
    )
    charge_spilling: list[float] = Field(description="Absolute charge spilling")
    total_spilling: list[float] = Field(description="Total spilling")
    elements: list[str] = Field(description="Elements in structure")
    basis_type: list[str] = Field(description="Basis set used in Lobster")
    basis_functions: list[list[str]] = Field(description="basis_functions")
    timing: dict[str, dict[str, str]] = Field(description="Dict with infos on timing")
    warning_lines: Optional[list] = Field(None, description="Warnings")
    info_orthonormalization: Optional[list] = Field(
        None, description="additional information on orthonormalization"
    )
    info_lines: Optional[list] = Field(
        None, description="list of strings with additional info lines"
    )
    has_doscar: Optional[bool] = Field(
        None, description="Bool indicating if DOSCAR is present."
    )
    has_doscar_lso: Optional[bool] = Field(
        None, description="Bool indicating if DOSCAR.LSO is present."
    )
    has_cohpcar: Optional[bool] = Field(
        None, description="Bool indicating if COHPCAR is present."
    )
    has_coopcar: Optional[bool] = Field(
        None, description="Bool indicating if COOPCAR is present."
    )
    has_cobicar: Optional[bool] = Field(
        None, description="Bool indicating if COBICAR is present."
    )
    has_charge: Optional[bool] = Field(
        None, description="Bool indicating if CHARGE is present."
    )
    has_madelung: Optional[bool] = Field(
        None,
        description="Bool indicating if Site Potentials and Madelung file is present.",
    )
    has_projection: Optional[bool] = Field(
        None, description="Bool indicating if projection file is present."
    )
    has_bandoverlaps: Optional[bool] = Field(
        None, description="Bool indicating if BANDOVERLAPS file is present"
    )
    has_fatbands: Optional[bool] = Field(
        None, description="Bool indicating if Fatbands are present."
    )
    has_grosspopulation: Optional[bool] = Field(
        None, description="Bool indicating if GROSSPOP file is present."
    )
    has_density_of_energies: Optional[bool] = Field(
        None, description="Bool indicating if DensityofEnergies is present"
    )


class LobsterinModel(BaseModel):
    """Definition of input settings for the LOBSTER computation."""

    cohpstartenergy: float = Field(description="Start energy for COHP computation")
    cohpendenergy: float = Field(description="End energy for COHP computation")

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
    basisset: str = Field(description="basis set of computation")
    cohpgenerator: str = Field(
        description="Build the list of atom pairs to be analyzed using given distance"
    )
    saveprojectiontofile: Optional[bool] = Field(
        None, description="Save the results of projections"
    )
    lsodos: Optional[bool] = Field(
        None, description="Writes DOS output from the orthonormalized LCAO basis"
    )
    basisfunctions: list[str] = Field(
        description="Specify the basis functions for element"
    )


class Bonding(BaseModel):
    """Model describing bonding field of BondsInfo."""

    integral: Optional[float] = Field(
        None, description="Integral considering only bonding contributions from COHPs"
    )
    perc: Optional[float] = Field(
        None, description="Percentage of bonding contribution"
    )


class Antibonding(BaseModel):
    """Model describing antibonding field of BondsInfo."""

    integral: Optional[float] = Field(
        None,
        description="Integral considering only anti-bonding contributions from COHPs",
    )
    perc: Optional[float] = Field(
        None, description="Percentage of anti-bonding contribution"
    )


class BondsInfo(BaseModel):
    """Model describing bonds field of SiteInfo."""

    ICOHP_mean: str = Field(..., description="Mean of ICOHPs of relevant bonds")
    ICOHP_sum: str = Field(..., description="Sum of ICOHPs of relevant bonds")
    has_antibdg_states_below_Efermi: bool = Field(  # noqa: N815
        ...,
        description="Indicates if antibonding interactions below efermi are detected",
    )
    number_of_bonds: int = Field(
        ..., description="Number of bonds considered in the analysis"
    )
    bonding: Bonding = Field(description="Model describing bonding contributions")
    antibonding: Antibonding = Field(
        description="Model describing anti-bonding contributions"
    )


class SiteInfo(BaseModel):
    """Outer model describing sites field of Sites model."""

    env: str = Field(
        ...,
        description="The coordination environment identified from "
        "the LobsterPy analysis",
    )
    bonds: dict[str, BondsInfo] = Field(
        ...,
        description="A dictionary with keys as atomic-specie as key "
        "and BondsInfo model as values",
    )
    ion: str = Field(..., description="Ion to which the atom is bonded")
    charge: float = Field(..., description="Mulliken charge of the atom")
    relevant_bonds: list[str] = Field(
        ...,
        description="List of bond labels from the LOBSTER files i.e. for e.g. "
        " from ICOHPLIST.lobster/ COHPCAR.lobster",
    )


class Sites(BaseModel):
    """Model describing, sites field of CondensedBondingAnalysis."""

    sites: dict[int, SiteInfo] = Field(
        ...,
        description="A dictionary with site index as keys and SiteInfo model as values",
    )


class CohpPlotData(BaseModel):
    """Model describing the cohp_plot_data field of CondensedBondingAnalysis."""

    data: dict[str, Cohp] = Field(
        ...,
        description="A dictionary with plot labels from LobsterPy "
        "automatic analysis as keys and Cohp objects as values",
    )


class DictIons(BaseModel):
    """Model describing final_dict_ions field of CondensedBondingAnalysis."""

    data: dict[str, dict[str, int]] = Field(
        ...,
        description="Dict consisting information on environments of cations "
        "and counts for them",
    )


class DictBonds(BaseModel):
    """Model describing final_dict_bonds field of CondensedBondingAnalysis."""

    data: dict[str, dict[str, Union[float, bool]]] = Field(
        ..., description="Dict consisting information on ICOHPs per bond type"
    )


class CondensedBondingAnalysis(BaseModel):
    """Definition of condensed bonding analysis data from LobsterPy ICOHP."""

    formula: str = Field(description="Pretty formula of the structure")
    max_considered_bond_length: float = Field(
        description="Maximum bond length considered in bonding analysis"
    )
    limit_icohp: list[Union[str, float]] = Field(
        description="ICOHP range considered in co-ordination environment analysis"
    )
    number_of_considered_ions: int = Field(
        ..., description="number of ions detected based on Mulliken/Löwdin Charges"
    )
    sites: Sites = Field(
        ...,
        description="Bonding information at inequivalent sites in the structure",
    )
    type_charges: str = Field(
        description="Charge type considered for assigning valences in bonding analysis"
    )
    cutoff_icohp: float = Field(
        description="Percent limiting the ICOHP values to be considered"
        " relative to strongest ICOHP",
    )
    summed_spins: bool = Field(
        description="Bool that states if the spin channels in the "
        "cohp_plot_data are summed.",
    )
    start: Optional[float] = Field(
        None,
        description="Sets the lower limit of energy relative to Fermi for evaluating"
        " bonding/anti-bonding percentages in the bond"
        " if set to None, all energies up-to the Fermi is considered",
    )
    cohp_plot_data: CohpPlotData = Field(
        ...,
        description="Plotting data for the relevant bonds from LobsterPy analysis",
    )
    which_bonds: str = Field(
        description="Specifies types of bond considered in LobsterPy analysis",
    )
    final_dict_bonds: DictBonds = Field(
        ...,
        description="Dict consisting information on ICOHPs per bond type",
    )
    final_dict_ions: DictIons = Field(
        ...,
        description="Model that describes final_dict_ions field",
    )
    run_time: float = Field(
        ..., description="Time needed to run Lobsterpy condensed bonding analysis"
    )

    @classmethod
    def from_directory(
        cls,
        dir_name: Union[str, Path],
        save_cohp_plots: bool = True,
        lobsterpy_kwargs: dict = None,
        plot_kwargs: dict = None,
        which_bonds: str = "all",
    ) -> tuple:
        """Create a task document from a directory containing LOBSTER files.

        Parameters
        ----------
        dir_name : path or str
            The path to the folder containing the calculation outputs.
        save_cohp_plots : bool.
            Bool to indicate whether automatic cohp plots and jsons
            from lobsterpy will be generated.
        lobsterpy_kwargs : dict.
            kwargs to change default lobsterpy automatic analysis parameters.
        plot_kwargs : dict.
            kwargs to change plotting options in lobsterpy.
        which_bonds: str.
            mode for condensed bonding analysis: "cation-anion" and "all".
        """
        plot_kwargs = plot_kwargs or {}
        lobsterpy_kwargs = lobsterpy_kwargs or {}
        dir_name = Path(dir_name)
        cohpcar_path = Path(zpath(dir_name / "COHPCAR.lobster"))
        charge_path = Path(zpath(dir_name / "CHARGE.lobster"))
        structure_path = Path(zpath(dir_name / "POSCAR"))
        icohplist_path = Path(zpath(dir_name / "ICOHPLIST.lobster"))
        icobilist_path = Path(zpath(dir_name / "ICOBILIST.lobster"))
        icooplist_path = Path(zpath(dir_name / "ICOOPLIST.lobster"))

        # Update lobsterpy analysis parameters with user supplied parameters
        lobsterpy_kwargs_updated = {
            "are_cobis": False,
            "are_coops": False,
            "cutoff_icohp": 0.10,
            "noise_cutoff": 0.1,
            "orbital_cutoff": 0.05,
            "orbital_resolved": False,
            "start": None,
            "summed_spins": False,  # we will always use spin polarization here
            "type_charge": None,
            **lobsterpy_kwargs,
        }

        try:
            start = time.time()
            analyse = Analysis(
                path_to_poscar=structure_path,
                path_to_icohplist=icohplist_path,
                path_to_cohpcar=cohpcar_path,
                path_to_charge=charge_path,
                which_bonds=which_bonds,
                **lobsterpy_kwargs_updated,
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
            limit_icohp_val = list(cba["limit_icohp"])
            _replace_inf_values(limit_icohp_val)

            condensed_bonding_analysis = CondensedBondingAnalysis(
                formula=cba["formula"],
                max_considered_bond_length=cba["max_considered_bond_length"],
                limit_icohp=limit_icohp_val,
                number_of_considered_ions=cba["number_of_considered_ions"],
                sites=Sites(**cba),
                type_charges=analyse.type_charge,
                cohp_plot_data=CohpPlotData(data=cba_cohp_plot_data),
                cutoff_icohp=analyse.cutoff_icohp,
                summed_spins=lobsterpy_kwargs_updated.get("summed_spins"),
                which_bonds=analyse.which_bonds,
                final_dict_bonds=DictBonds(data=analyse.final_dict_bonds),
                final_dict_ions=DictIons(data=analyse.final_dict_ions),
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

                filename = dir_name / f"condensed_bonding_analysis_{which_bonds}"
                with open(f"{filename}.json", "w") as fp:
                    json.dump(analyse.condensed_bonding_analysis, fp)
                with open(f"{filename}.txt", "w") as fp:
                    for line in describe.text:
                        fp.write(f"{line}\n")

            # Read in strongest icohp values
            sb = _identify_strongest_bonds(
                analyse=analyse,
                icobilist_path=icobilist_path,
                icohplist_path=icohplist_path,
                icooplist_path=icooplist_path,
            )

        except ValueError:
            return None, None, None
        else:
            return condensed_bonding_analysis, describe, sb


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
    tanimoto_summed: Optional[float] = Field(
        None,
        description="Tanimoto similarity index for summed PDOS between "
        "VASP and LOBSTER",
    )
    e_range: list[Union[float, None]] = Field(
        description="Energy range used for evaluating the Tanimoto similarity index"
    )
    n_bins: Optional[int] = Field(
        None,
        description="Number of bins used for discretizing the VASP and LOBSTER PDOS"
        "(Affects the Tanimoto similarity index)",
    )


class ChargeComparisons(BaseModel):
    """Model describing the charges field in the CalcQualitySummary model."""

    bva_mulliken_agree: Optional[bool] = Field(
        None,
        description="Bool indicating whether atoms classification as cation "
        "or anion based on Mulliken charge signs of LOBSTER "
        "agree with BVA analysis",
    )
    bva_loewdin_agree: Optional[bool] = Field(
        None,
        description="Bool indicating whether atoms classification as cations "
        "or anions based on Loewdin charge signs of LOBSTER "
        "agree with BVA analysis",
    )


class BandOverlapsComparisons(BaseModel):
    """Model describing the Band overlaps field in the CalcQualitySummary model."""

    file_exists: bool = Field(
        description="Boolean indicating whether the bandOverlaps.lobster "
        "file is generated during the LOBSTER run",
    )
    limit_maxDeviation: Optional[float] = Field(  # noqa: N815
        None,
        description="Limit set for maximal deviation in pymatgen parser",
    )
    has_good_quality_maxDeviation: Optional[bool] = Field(  # noqa: N815
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
        description="Absolute charge spilling value from the LOBSTER calculation.",
    )
    abs_total_spilling: float = Field(
        description="Total charge spilling percent from the LOBSTER calculation.",
    )


class CalcQualitySummary(BaseModel):
    """Model describing the calculation quality of lobster run."""

    minimal_basis: bool = Field(
        description="Denotes whether the calculation used the minimal basis for the "
        "LOBSTER computation",
    )
    charge_spilling: ChargeSpilling = Field(
        description="Model describing the charge spilling from the LOBSTER runs",
    )
    charge_comparisons: Optional[ChargeComparisons] = Field(
        None,
        description="Model describing the charge sign comparison results",
    )
    band_overlaps_analysis: Optional[BandOverlapsComparisons] = Field(
        None,
        description="Model describing the band overlap file analysis results",
    )
    dos_comparisons: Optional[DosComparisons] = Field(
        None,
        description="Model describing the VASP and LOBSTER PDOS comparisons results",
    )

    @classmethod
    @requires(Analysis, "lobsterpy must be installed to create an CalcQualitySummary.")
    def from_directory(
        cls,
        dir_name: Union[Path, str],
        calc_quality_kwargs: dict = None,
    ) -> Self:
        """Make a LOBSTER calculation quality summary from directory with LOBSTER files.

        Parameters
        ----------
        dir_name : path or str
            The path to the folder containing the calculation outputs.
        calc_quality_kwargs : dict
            kwargs to change calc quality analysis options in lobsterpy.

        Returns
        -------
        CalcQualitySummary
            A task document summarizing quality of the lobster calculation.
        """
        dir_name = Path(dir_name)
        calc_quality_kwargs = calc_quality_kwargs or {}
        band_overlaps_path = Path(zpath(dir_name / "bandOverlaps.lobster"))
        charge_path = Path(zpath(dir_name / "CHARGE.lobster"))
        doscar_path = Path(
            zpath(
                dir_name / "DOSCAR.LSO.lobster"
                if Path(zpath(dir_name / "DOSCAR.LSO.lobster")).exists()
                else Path(zpath(dir_name / "DOSCAR.lobster"))
            )
        )
        lobsterin_path = Path(zpath(dir_name / "lobsterin"))
        lobsterout_path = Path(zpath(dir_name / "lobsterout"))
        potcar_path = (
            Path(zpath(dir_name / "POTCAR"))
            if Path(zpath(dir_name / "POTCAR")).exists()
            else None
        )
        structure_path = Path(zpath(dir_name / "POSCAR"))
        vasprun_path = Path(zpath(dir_name / "vasprun.xml"))

        # Update calc quality kwargs supplied by user
        calc_quality_kwargs_updated = {
            "e_range": [-20, 0],
            "dos_comparison": True,
            "n_bins": 256,
            "bva_comp": True,
            **calc_quality_kwargs,
        }
        cal_quality_dict = Analysis.get_lobster_calc_quality_summary(
            path_to_poscar=structure_path,
            path_to_vasprun=vasprun_path,
            path_to_charge=charge_path,
            path_to_potcar=potcar_path,
            path_to_doscar=doscar_path,
            path_to_lobsterin=lobsterin_path,
            path_to_lobsterout=lobsterout_path,
            path_to_bandoverlaps=band_overlaps_path,
            **calc_quality_kwargs_updated,
        )
        return CalcQualitySummary(**cal_quality_dict)


class StrongestBonds(BaseModel):
    """Strongest bonds extracted from ICOHPLIST/ICOOPLIST/ICOBILIST from LOBSTER.

    LobsterPy is used for the extraction.
    """

    which_bonds: Optional[str] = Field(
        None,
        description="Denotes whether the information "
        "is for cation-anion pairs or all bonds",
    )
    strongest_bonds_icoop: Optional[dict] = Field(
        None,
        description="Dict with infos on bond strength and bond length based on ICOOP.",
    )
    strongest_bonds_icohp: Optional[dict] = Field(
        None,
        description="Dict with infos on bond strength and bond length based on ICOHP.",
    )
    strongest_bonds_icobi: Optional[dict] = Field(
        None,
        description="Dict with infos on bond strength and bond length based on ICOBI.",
    )


class LobsterTaskDocument(StructureMetadata, extra="allow"):  # type: ignore[call-arg]
    """Definition of LOBSTER task document."""

    structure: Structure = Field(description="The structure used in this task")
    dir_name: Union[str, Path] = Field(
        description="The directory for this Lobster task"
    )
    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp for this task document was last updated",
    )
    charges: Optional[Charge] = Field(
        None,
        description="pymatgen Charge obj. Contains atomic charges based on Mulliken "
        "and Loewdin charge analysis",
    )
    lobsterout: LobsteroutModel = Field(description="Lobster out data")
    lobsterin: LobsterinModel = Field(description="Lobster calculation inputs")
    lobsterpy_data: Optional[CondensedBondingAnalysis] = Field(
        None, description="Model describing the LobsterPy data"
    )
    lobsterpy_text: Optional[str] = Field(
        None, description="Stores LobsterPy automatic analysis summary text"
    )
    calc_quality_summary: Optional[CalcQualitySummary] = Field(
        None,
        description="Model summarizing results of lobster runs like charge spillings, "
        "band overlaps, DOS comparisons with VASP runs and quantum chemical LOBSTER "
        "charge sign comparisons with BVA method",
    )
    calc_quality_text: Optional[str] = Field(
        None, description="Stores calculation quality analysis summary text"
    )
    strongest_bonds: Optional[StrongestBonds] = Field(
        None,
        description="Describes the strongest cation-anion ICOOP, ICOBI and ICOHP bonds",
    )
    lobsterpy_data_cation_anion: Optional[CondensedBondingAnalysis] = Field(
        None, description="Model describing the LobsterPy data"
    )
    lobsterpy_text_cation_anion: Optional[str] = Field(
        None,
        description="Stores LobsterPy automatic analysis summary text",
    )
    strongest_bonds_cation_anion: Optional[StrongestBonds] = Field(
        None,
        description="Describes the strongest cation-anion ICOOP, ICOBI and ICOHP bonds",
    )
    dos: Optional[LobsterCompleteDos] = Field(
        None, description="pymatgen pymatgen.io.lobster.Doscar.completedos data"
    )
    lso_dos: Optional[LobsterCompleteDos] = Field(
        None, description="pymatgen pymatgen.io.lobster.Doscar.completedos data"
    )
    madelung_energies: Optional[MadelungEnergies] = Field(
        None,
        description="pymatgen Madelung energies obj. Contains madelung energies"
        "based on Mulliken and Loewdin charges",
    )
    site_potentials: Optional[SitePotential] = Field(
        None,
        description="pymatgen Site potentials obj. Contains site potentials "
        "based on Mulliken and Loewdin charges",
    )
    gross_populations: Optional[Grosspop] = Field(
        None,
        description="pymatgen Grosspopulations obj. Contains gross populations "
        " based on Mulliken and Loewdin charges ",
    )
    band_overlaps: Optional[Bandoverlaps] = Field(
        None,
        description="pymatgen Bandoverlaps obj for each k-point from"
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
    icohp_list: Optional[Icohplist] = Field(
        None, description="pymatgen Icohplist object with ICOHP data"
    )
    icoop_list: Optional[Icohplist] = Field(
        None, description="pymatgen Icohplist object with ICOOP data"
    )
    icobi_list: Optional[Icohplist] = Field(
        None, description="pymatgen Icohplist object with ICOBI data"
    )

    schema: str = Field(
        __version__, description="Version of atomate2 used to create the document"
    )

    @classmethod
    @requires(
        Analysis,
        "LobsterTaskDocument requires `lobsterpy` and `ijson` to function properly. "
        "Please reinstall atomate2 using atomate2[lobster]",
    )
    def from_directory(
        cls,
        dir_name: Union[Path, str],
        additional_fields: dict = None,
        add_coxxcar_to_task_document: bool = False,
        analyze_outputs: bool = True,
        calc_quality_kwargs: dict = None,
        lobsterpy_kwargs: dict = None,
        plot_kwargs: dict = None,
        store_lso_dos: bool = False,
        save_cohp_plots: bool = True,
        save_cba_jsons: bool = True,
        save_computational_data_jsons: bool = False,
    ) -> Self:
        """Create a task document from a directory containing LOBSTER files.

        Parameters
        ----------
        dir_name : path or str.
            The path to the folder containing the calculation outputs.
        additional_fields : dict.
            Dictionary of additional fields to add to output document.
        add_coxxcar_to_task_document : bool.
            Bool to indicate whether to add COHPCAR, COOPCAR, COBICAR data objects
            to the task document.
        analyze_outputs: bool.
            If True, will enable lobsterpy analysis.
        calc_quality_kwargs : dict.
            kwargs to change calc quality summary options in lobsterpy.
        lobsterpy_kwargs : dict.
            kwargs to change default lobsterpy automatic analysis parameters.
        plot_kwargs : dict.
            kwargs to change plotting options in lobsterpy.
        store_lso_dos : bool.
            Whether to store the LSO DOS.
        save_cohp_plots : bool.
            Bool to indicate whether automatic cohp plots and jsons
            from lobsterpy will be generated.
        save_cba_jsons : bool.
            Bool to indicate whether condensed bonding analysis jsons
            should be saved, consists of outputs from lobsterpy analysis,
            calculation quality summary, lobster dos, charges and madelung energies
        save_computational_data_jsons : bool.
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
        lobsterout_doc = Lobsterout(Path(zpath(dir_name / "lobsterout"))).get_doc()
        lobster_out = LobsteroutModel(**lobsterout_doc)
        lobster_in = LobsterinModel(
            **Lobsterin.from_file(Path(zpath(dir_name / "lobsterin")))
        )

        icohplist_path = Path(zpath(dir_name / "ICOHPLIST.lobster"))
        icooplist_path = Path(zpath(dir_name / "ICOOPLIST.lobster"))
        icobilist_path = Path(zpath(dir_name / "ICOBILIST.lobster"))
        cohpcar_path = Path(zpath(dir_name / "COHPCAR.lobster"))
        charge_path = Path(zpath(dir_name / "CHARGE.lobster"))
        cobicar_path = Path(zpath(dir_name / "COBICAR.lobster"))
        coopcar_path = Path(zpath(dir_name / "COOPCAR.lobster"))
        doscar_path = Path(zpath(dir_name / "DOSCAR.lobster"))
        structure_path = Path(zpath(dir_name / "POSCAR"))
        madelung_energies_path = Path(zpath(dir_name / "MadelungEnergies.lobster"))
        site_potentials_path = Path(zpath(dir_name / "SitePotentials.lobster"))
        gross_populations_path = Path(zpath(dir_name / "GROSSPOP.lobster"))
        band_overlaps_path = Path(zpath(dir_name / "bandOverlaps.lobster"))

        icohp_list = icoop_list = icobi_list = None
        if icohplist_path.exists():
            icohp_list = Icohplist(filename=icohplist_path)
        if icooplist_path.exists():
            icoop_list = Icohplist(filename=icooplist_path, are_coops=True)
        if icobilist_path.exists():
            icobi_list = Icohplist(filename=icobilist_path, are_cobis=True)

        # Do automatic bonding analysis with LobsterPy
        struct = Structure.from_file(structure_path)

        # will perform two condensed bonding analysis computations
        condensed_bonding_analysis = None
        condensed_bonding_analysis_ionic = None
        sb_all = None
        sb_ionic = None
        calc_quality_summary = None
        calc_quality_text = None
        describe = None
        describe_ionic = None
        if analyze_outputs:
            if (
                icohplist_path.exists()
                and cohpcar_path.exists()
                and charge_path.exists()
            ):
                (
                    condensed_bonding_analysis,
                    describe,
                    sb_all,
                ) = CondensedBondingAnalysis.from_directory(
                    dir_name,
                    save_cohp_plots=save_cohp_plots,
                    plot_kwargs=plot_kwargs,
                    lobsterpy_kwargs=lobsterpy_kwargs,
                    which_bonds="all",
                )
                (
                    condensed_bonding_analysis_ionic,
                    describe_ionic,
                    sb_ionic,
                ) = CondensedBondingAnalysis.from_directory(
                    dir_name,
                    save_cohp_plots=save_cohp_plots,
                    plot_kwargs=plot_kwargs,
                    lobsterpy_kwargs=lobsterpy_kwargs,
                    which_bonds="cation-anion",
                )
            # Get lobster calculation quality summary data

            calc_quality_summary = CalcQualitySummary.from_directory(
                dir_name,
                calc_quality_kwargs=calc_quality_kwargs,
            )

            calc_quality_text = Description.get_calc_quality_description(
                calc_quality_summary.model_dump()
            )

        # Read in charges
        charges = None
        if charge_path.exists():
            charges = Charge(filename=charge_path)

        # Read in DOS
        dos = None
        if doscar_path.exists():
            doscar_lobster = Doscar(doscar=doscar_path, structure_file=structure_path)
            dos = doscar_lobster.completedos

        # Read in LSO DOS
        lso_dos = None
        doscar_lso_path = Path(zpath(dir_name / "DOSCAR.LSO.lobster"))
        if store_lso_dos and doscar_lso_path.exists():
            doscar_lso_lobster = Doscar(
                doscar=doscar_lso_path, structure_file=structure_path
            )
            lso_dos = doscar_lso_lobster.completedos

        # Read in Madelung energies
        madelung_energies = None
        if madelung_energies_path.exists():
            madelung_energies = MadelungEnergies(filename=madelung_energies_path)

        # Read in Site Potentials
        site_potentials = None
        if site_potentials_path.exists():
            site_potentials = SitePotential(filename=site_potentials_path)

        # Read in Gross Populations
        gross_populations = None
        if gross_populations_path.exists():
            gross_populations = Grosspop(filename=gross_populations_path)

        # Read in Band overlaps
        band_overlaps = None
        if band_overlaps_path.exists():
            band_overlaps = Bandoverlaps(filename=band_overlaps_path)

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
            strongest_bonds=sb_all,
            lobsterpy_data_cation_anion=condensed_bonding_analysis_ionic,
            lobsterpy_text_cation_anion=" ".join(describe_ionic.text)
            if describe_ionic is not None
            else None,
            strongest_bonds_cation_anion=sb_ionic,
            calc_quality_summary=calc_quality_summary,
            calc_quality_text=" ".join(calc_quality_text)
            if calc_quality_text is not None
            else None,
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
            icohp_list=icohp_list,
            icoop_list=icoop_list,
            icobi_list=icobi_list,
        )

        if save_cba_jsons and analyze_outputs:
            cba_json_save_dir = dir_name / "cba.json.gz"
            with gzip.open(cba_json_save_dir, "wt", encoding="UTF-8") as file:
                # Write the json in iterable format
                # (Necessary to load large JSON files via ijson)
                file.write("[")
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
                            "strongest_bonds": doc.strongest_bonds_cation_anion,
                        }
                    }
                else:
                    data = {"cation_anion_bonds": {}}
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, file)
                file.write(",")
                # add all-bonds data
                lobsterpy_analysis_type = doc.lobsterpy_data.which_bonds
                data = {
                    f"{lobsterpy_analysis_type}_bonds": {
                        "lobsterpy_data": doc.lobsterpy_data,
                        "lobsterpy_text": ["".join(doc.lobsterpy_text)],
                        "strongest_bonds": doc.strongest_bonds,
                    }
                }
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, file)
                file.write(",")
                data = {
                    "madelung_energies": doc.madelung_energies
                }  # add madelung energies
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, file)
                file.write(",")
                data = {"charges": doc.charges}  # add charges
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, file)
                file.write(",")
                data = {
                    "calc_quality_summary": doc.calc_quality_summary
                }  # add calc quality summary dict
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, file)
                file.write(",")
                data = {
                    "calc_quality_text": ["".join(doc.calc_quality_text)]  # type: ignore[dict-item]
                }  # add calc quality summary dict
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, file)
                file.write(",")
                data = {"dos": doc.dos}  # add NON LSO of lobster
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, file)
                file.write(",")
                data = {"lso_dos": doc.lso_dos}  # add LSO DOS of lobster
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=True, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, file)
                file.write(",")
                data = {"builder_meta": doc.builder_meta}  # add builder metadata
                monty_encoded_json_doc = jsanitize(
                    data, allow_bson=False, strict=True, enum_values=True
                )
                json.dump(monty_encoded_json_doc, file)
                del data, monty_encoded_json_doc
                file.write("]")

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
                doc.cohp_data = cohp_obj

            if coopcar_path.exists() and doc.coop_data is None:
                coop_obj = CompleteCohp.from_file(
                    fmt="LOBSTER",
                    structure_file=structure_path,
                    filename=coopcar_path,
                    are_coops=True,
                    are_cobis=False,
                )
                doc.coop_data = coop_obj

            if cobicar_path.exists() and doc.cobi_data is None:
                cobi_obj = CompleteCohp.from_file(
                    fmt="LOBSTER",
                    structure_file=structure_path,
                    filename=cobicar_path,
                    are_coops=False,
                    are_cobis=True,
                )
                doc.cobi_data = cobi_obj
            with gzip.open(
                computational_data_json_save_dir, "wt", encoding="UTF-8"
            ) as file:
                # Write the json in iterable format
                # (Necessary to load large JSON files via ijson)
                file.write("[")
                for attribute in doc.model_fields:
                    if attribute not in fields_to_exclude:
                        # Use monty encoder to automatically convert pymatgen
                        # objects and other data json compatible dict format
                        data = {
                            attribute: jsanitize(
                                getattr(doc, attribute),
                                allow_bson=False,
                                strict=True,
                                enum_values=True,
                            )
                        }
                        json.dump(data, file)
                        if attribute != list(doc.model_fields.keys())[-1]:
                            file.write(",")  # add comma separator between two dicts
                        del data
                file.write("]")

            # Again unset the cohp, cobi and coop data fields if not desired in the DB
            if not add_coxxcar_to_task_document:
                doc.cohp_data = None
                doc.coop_data = None
                doc.cobi_data = None

        return doc.model_copy(update=additional_fields)


def _replace_inf_values(data: Union[dict[Any, Any], list[Any]]) -> None:
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
) -> StrongestBonds:
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
    StrongestBonds
    """
    data = [
        (icohplist_path, False, False, "icohp"),
        (icobilist_path, True, False, "icobi"),
        (icooplist_path, False, True, "icoop"),
    ]
    output = []
    model_data = {"which_bonds": analyse.which_bonds}
    for file, are_cobis, are_coops, prop in data:
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
            model_data[f"strongest_bonds_{prop}"] = bond_dict
            output.append(
                StrongestBonds(
                    strongest_bonds=bond_dict,
                    which_bonds=analyse.which_bonds,
                )
            )
        else:
            model_data[f"strongest_bonds_{prop}"] = {}
            output.append(None)
    return StrongestBonds(**model_data)


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

    for idx, val in enumerate(bond_labels_unique):
        for j, val2 in enumerate(bonds):
            if val == val2:
                sep_icohp[idx].append(icohp_all[j])
                sep_lengths[idx].append(lengths[j])

    if are_cobis and not are_coops:
        prop = "icobi"
    elif not are_cobis and are_coops:
        prop = "icoop"
    else:
        prop = "icohp"

    bond_dict = {}
    for idx, lab in enumerate(bond_labels_unique):
        label = lab.split("-")
        label.sort()
        for rel_bnd in relevant_bonds:
            rel_bnd_list = rel_bnd.split("-")
            rel_bnd_list.sort()
            if label == rel_bnd_list:
                if prop == "icohp":
                    index = np.argmin(sep_icohp[idx])
                    bond_dict.update(
                        {
                            rel_bnd: {
                                "bond_strength": min(sep_icohp[idx]),
                                "length": sep_lengths[idx][index],
                            }
                        }
                    )
                else:
                    index = np.argmax(sep_icohp[idx])
                    bond_dict.update(
                        {
                            rel_bnd: {
                                "bond_strength": max(sep_icohp[idx]),
                                "length": sep_lengths[idx][index],
                            }
                        }
                    )
    return bond_dict


def read_saved_json(
    filename: str, pymatgen_objs: bool = True, query: str = "structure"
) -> dict[str, Any]:
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
    with gzip.open(filename, "rb") as file:
        lobster_data = {
            field: data
            for obj in ijson.items(file, "item", use_float=True)
            for field, data in obj.items()
            if query is None or query in obj
        }
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
            elif "lobsterpy_data" in query_key:
                for field in lobster_data[query_key].__fields__:
                    val = MontyDecoder().process_decoded(
                        getattr(lobster_data[query_key], field)
                    )
                    setattr(lobster_data[query_key], field, val)

    return lobster_data
