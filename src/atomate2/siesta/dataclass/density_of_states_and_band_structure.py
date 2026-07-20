"""
Data class managing DOS and band-structure analysis options for SIESTA input.

class DensityOfStatesAndBandStructure

Based on User's Guide Siesta 5.4.0
Section: 6.16 Band-structure analysis
         6.16.1 Format of the .bands file
         6.16.2 Output of wavefunctions associated to bands
         6.17 Output of selected wavefunctions
         6.18 Density of states
         6.18.1 Total density of states
         6.18.2 Partial (projected) density of states
         6.18.3 Local density of states
         6.19 Options for chemical analysis
         6.19.1 Mulliken charges and overlap populations
         6.19.2 Depreceted population flags (Voronoi and Hirshfeld atomic
                population analysis)
         6.19.3 Crystal-Orbital overlap and hamilton populations (COOP/COHP)

"""

# Metadata

__all__ = ["DensityOfStatesAndBandStructure"]

import logging
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, cast

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.sets.bands import band_paymatgen_to_siesta
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class DensityOfStatesAndBandStructure(FDFDataclass):
    """Manage density of states (DOS) and band-structure analysis options.

    Data class covering SIESTA User's Guide sections 6.16-6.19 (band structure,
    wavefunctions, density of states, and chemical-analysis population flags).
    """

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.ERROR
    )  # Default to show errors only

    # Default changed to "ReciprocalLatticeVectors" to match pymatgen's band path
    # generation
    # ----------------------------
    # 6.16 Band-structure analysis
    # ----------------------------
    # band_line_scale: str = "pi/a" # BandLinesScale pi/a
    # band_lines_block: Optional [Dict[float,Any]]= field(default_factory=dict) #
    # %block BandLines 〈None〉
    # band_points_block: Optional [Dict[float,Any]]= field(default_factory=dict) #
    # %block BandPoints 〈None〉
    # write_k_bands: bool = False # WriteKbands false
    # write_bands: bool = False # WriteBands  false
    band_line_scale: str = field(
        default="ReciprocalLatticeVectors",
        metadata={
            "description": (
                "Sets the scale for the k-point coordinates used in the 'BandLines' "
                "block. Common options are 'pi/a' or 'ReciprocalLatticeVectors'."
            ),
            "SIESTA keyword": "BandLinesScale",
        },
    )

    band_lines_block: dict[float, Any] | None = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to define the high-symmetry lines (paths) in the Brillouin "
                "zone along which the electronic band structure will be calculated."
            ),
            "SIESTA keyword": "%block BandLines",
        },
    )

    band_points_block: dict[float, Any] | None = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to define a set of individual, discrete k-points at which the "
                "electronic bands will be calculated, as an alternative to defining "
                "lines."
            ),
            "SIESTA keyword": "%block BandPoints",
        },
    )

    write_k_bands: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, writes the coordinates and accumulated distances of the "
                "k-points along the band structure path to a file, which is useful for "
                "plotting."
            ),
            "SIESTA keyword": "WriteKbands",
        },
    )

    write_bands: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, writes the calculated eigenvalues (the bands) along the "
                "specified k-point path to the output file for plotting."
            ),
            "SIESTA keyword": "WriteBands",
        },
    )

    # --------------------------------------------------
    # 6.16.1 Format of the .bands file
    # 6.16.2 Output of wavefunctions associated to bands
    # --------------------------------------------------
    # wfs_write_for_bands: bool = False  #WFS.Write.For.Bands false
    # wfs_band_min: int = 1 # WFS.Band.Min 1
    # wfs_band_max: int = None # WFS.Band.Max number of orbitals
    wfs_write_for_bands: bool = field(
        default=False,
        metadata={
            "description": (
                "A flag to enable writing the wavefunctions for each state calculated "
                "along the band structure path."
            ),
            "SIESTA keyword": "WFS.Write.For.Bands",
        },
    )

    wfs_band_min: int = field(
        default=1,
        metadata={
            "description": (
                "The minimum band index for which the wavefunction will be written."
            ),
            "SIESTA keyword": "WFS.Band.Min",
        },
    )

    wfs_band_max: int | None = field(
        default=None,
        metadata={
            "description": (
                "The maximum band index for which the wavefunction will be written. "
                "Defaults to the total number of calculated orbitals if not set."
            ),
            "SIESTA keyword": "WFS.Band.Max",
        },
    )

    # -------------------------------------
    # 6.17 Output of selected wavefunctions
    # -------------------------------------
    # wave_func_k_point_scale: str = "ReciprocalLatticeVectors" # WaveFuncKPointsScale
    # pi/a
    # wave_func_k_points_block: Optional [Dict[float,Any]]= field(default_factory=dict)
    # # %block WaveFuncKPoints 〈None〉
    # write_wave_functions: bool = False # WriteWaveFunctions false
    wave_func_k_point_scale: str = field(
        default="ReciprocalLatticeVectors",
        metadata={
            "description": (
                "Sets the scale and basis for the k-point coordinates in the "
                "'WaveFuncKPoints' block. Common options are "
                "'ReciprocalLatticeVectors' (fractional) or 'pi/a'."
            ),
            "SIESTA keyword": "WaveFuncKPointsScale",
        },
    )

    # wave_func_k_points_block: Optional[Dict[float, Any]] = field(
    wave_func_k_points_block: dict[str, Any] | None = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to define a list of specific k-points at which the real-space "
                "wavefunctions will be calculated and written to output files."
            ),
            "SIESTA keyword": "%block WaveFuncKPoints",
        },
    )

    write_wave_functions: bool = field(
        default=False,
        metadata={
            "description": (
                "A master flag to enable the writing of real-space wavefunctions at "
                "the specific k-points defined in the 'WaveFuncKPoints' block."
            ),
            "SIESTA keyword": "WriteWaveFunctions",
        },
    )

    # ------------------------------
    # 6.18 Density of states
    # 6.18.1 Total density of states
    # ------------------------------
    # dos_kgrid_monkhorst_pack_block: Dict[float,Any]= field(default_factory=dict)  #
    # DOS.kgrid.MonkhorstPack
    # dos_kgrid_cutoff: float = None  # DOS.kgrid.Cutoff
    # dos_kgrid_file: str = None # DOS.kgrid.File
    dos_kgrid_monkhorst_pack_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to define a specific Monkhorst-Pack k-point grid to be used "
                "for the Density of States calculation, which can be denser than the "
                "SCF grid."
            ),
            "SIESTA keyword": "%block DOS.kgrid.MonkhorstPack",
        },
    )

    dos_kgrid_cutoff: float | None = field(
        default=None,
        metadata={
            "description": (
                "A real-space cutoff (in Angstroms) used to automatically generate a "
                "k-point grid specifically for the DOS calculation."
            ),
            "SIESTA keyword": "DOS.kgrid.Cutoff",
        },
    )

    dos_kgrid_file: str | None = field(
        default=None,
        metadata={
            "description": (
                "The name of a file from which to read the k-points to be used for the "
                "DOS calculation, offering maximum flexibility."
            ),
            "SIESTA keyword": "DOS.kgrid.File",
        },
    )

    # ---------------------------------------------
    # 6.18.2 Partial (projected) density of states
    # ---------------------------------------------
    # projected_density_of_states_block: Dict[float,Any]= field(default_factory=dict) #
    # %block ProjectedDensityOfStates 〈None〉
    # pdos_kgrid_monkhorst_pack_block: Dict[float,Any]= field(default_factory=dict) #
    # PDOS.kgrid.MonkhorstPack
    # pdos_kgrid_cutoff: float = None # PDOS.kgrid.Cutoff
    # pdos_kgrid_file: str = None # PDOS.kgrid.File
    projected_density_of_states_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to define the projections of the Density of States onto "
                "specific atomic orbitals, allowing analysis of orbital contributions."
            ),
            "SIESTA keyword": "%block ProjectedDensityOfStates",
        },
    )

    pdos_kgrid_monkhorst_pack_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to define a specific Monkhorst-Pack k-point grid to be used "
                "for the Projected Density of States (PDOS) calculation."
            ),
            "SIESTA keyword": "%block PDOS.kgrid.MonkhorstPack",
        },
    )

    pdos_kgrid_cutoff: float | None = field(
        default=None,
        metadata={
            "description": (
                "A real-space cutoff (in Angstroms) used to automatically generate a "
                "k-point grid specifically for the PDOS calculation."
            ),
            "SIESTA keyword": "PDOS.kgrid.Cutoff",
        },
    )

    pdos_kgrid_file: str | None = field(
        default=None,
        metadata={
            "description": (
                "The name of a file from which to read the k-points to be used for the "
                "PDOS calculation."
            ),
            "SIESTA keyword": "PDOS.kgrid.File",
        },
    )

    # -------------------------------
    # 6.18.3 Local density of states
    # -------------------------------
    # local_density_of_states_block: Dict[float,Any]= field(default_factory=dict) #
    # %block LocalDensityOfStates 〈None〉
    # ldos_kgrid_monkhorst_pack_block: Dict[float,Any]= field(default_factory=dict) #
    # LDOS.kgrid.MonkhorstPack
    # ldos_kgrid_cutoff: float = None # LDOS.kgrid.Cutoff
    # ldos_kgrid_file: str = None # LDOS.kgrid.File
    local_density_of_states_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to define the energy window (Emin, Emax) for which the "
                "real-space Local Density of States (LDOS) will be calculated."
            ),
            "SIESTA keyword": "%block LocalDensityOfStates",
        },
    )

    ldos_kgrid_monkhorst_pack_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to define a specific Monkhorst-Pack k-point grid to be used "
                "for the Local Density of States calculation."
            ),
            "SIESTA keyword": "%block LDOS.kgrid_Monkhorst_pack",
        },
    )

    ldos_kgrid_cutoff: float | None = field(
        default=None,
        metadata={
            "description": (
                "A real-space cutoff (in Angstroms) used to automatically generate a "
                "k-point grid specifically for the LDOS calculation."
            ),
            "SIESTA keyword": "LDOS.kgrid.Cutoff",
        },
    )

    ldos_kgrid_file: str | None = field(
        default=None,
        metadata={
            "description": (
                "The name of a file from which to read the k-points to be used for the "
                "LDOS calculation."
            ),
            "SIESTA keyword": "LDOS.kgrid.File",
        },
    )

    # ------------------------------------------------
    # 6.19 Options for chemical analysis
    # 6.19.1 Mulliken charges and overlap populations
    # ------------------------------------------------
    # write_mullkin_pop: int = 0 # WriteMullikenPop 0
    # mulliken_in_scf: bool = False # MullikenInSCF false
    # spin_in_scf: bool = True # SpinInSCF true
    write_mullkin_pop: int = field(
        default=0,
        metadata={
            "description": (
                "Sets the level of detail for the Mulliken population analysis. "
                "0=none, 1=per-atom, 2=per-orbital, 3=overlap populations."
            ),
            "SIESTA keyword": "WriteMullikenPop",
        },
    )

    mulliken_in_scf: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, performs and prints the Mulliken population analysis at "
                "every SCF iteration, not just at the end."
            ),
            "SIESTA keyword": "MullikenInSCF",
        },
    )

    spin_in_scf: bool = field(
        default=True,
        metadata={
            "description": (
                "If true, prints the integrated total and absolute spin polarization "
                "at every SCF iteration."
            ),
            "SIESTA keyword": "SpinInSCF",
        },
    )

    # --------------------------------------------------------
    # 6.19.2 (Deprecated population flag) Voronoi and Hirshfeld atomic population
    # analysis
    # --------------------------------------------------------
    # write_mulliken_pop:int = 0 #
    # write_harishfeld_pop: bool = False #Write.HirshfeldPop false
    # write_voronoi_pop: bool = False #Write.VoronoiPop false
    # partial_charges_at_every_geometry: bool = False #PartialChargesAtEveryGeometry
    # false
    # partial_charges_at_every_scf_step: bool = False #PartialChargesAtEverySCFStep
    # false
    # TODO:Write.MullikenPop
    write_mulliken_pop: int = field(
        default=0,
        metadata={
            "description": " ",
            "SIESTA keyword": "Write.MullikenPop",
        },
    )

    write_hirshfeld_pop: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, calculates and prints atomic partial charges using the "
                "Hirshfeld partitioning scheme."
            ),
            "SIESTA keyword": "Write.HirshfeldPop",
        },
    )

    write_voronoi_pop: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, calculates and prints atomic partial charges by integrating "
                "the charge density within each atom's Voronoi cell."
            ),
            "SIESTA keyword": "Write.VoronoiPop",
        },
    )

    partial_charges_at_every_geometry: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, calculates and prints partial charges at every step of a "
                "geometry optimization or molecular dynamics run."
            ),
            "SIESTA keyword": "PartialChargesAtEveryGeometry",
        },
    )

    partial_charges_at_every_scf_step: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, calculates and prints partial charges at every SCF iteration."
            ),
            "SIESTA keyword": "PartialChargesAtEverySCFStep",
        },
    )

    # -------------------------------------------------------------------
    # 6.19.3 Crystal-Orbital overlap and hamilton populations (COOP/COHP)
    # -------------------------------------------------------------------
    # coop_write: bool = False # COOP.Write false
    # wfs_energy_min: float = None # WFS.Energy.Min −∞  # noqa: RUF003
    # wfs_energy_max: float = None # WFS.Energy.Max ∞
    coop_write: bool = field(
        default=False,
        metadata={
            "description": (
                "A flag to enable the writing of data for Crystal Orbital Overlap "
                "Population (COOP) analysis, used to analyze chemical bonding."
            ),
            "SIESTA keyword": "COOP.Write",
        },
    )

    wfs_energy_min: float | None = field(
        default=None,
        metadata={
            "description": (
                "Sets a minimum energy threshold for writing wavefunctions. Only "
                "states with an energy above this value will be written."
            ),
            "SIESTA keyword": "WFS.Energy.Min",
        },
    )

    wfs_energy_max: float | None = field(
        default=None,
        metadata={
            "description": (
                "Sets a maximum energy threshold for writing wavefunctions. Only "
                "states with an energy below this value will be written."
            ),
            "SIESTA keyword": "WFS.Energy.Max",
        },
    )

    # calculate_total_dos: bool = True  # Flag to indicate if total density of states
    # (DOS) should be calculated
    # calculate_partial_dos: bool = False  # Flag to indicate if partial DOS should be
    # calculated
    # projected_dos_atoms: List[int] = field(default_factory=list)  # List of atom
    # indices for projected DOS (if any)
    # band_structure_kpoints: List[List[float]] = field(default_factory=list)  # List
    # of k-points for band structure calculation
    # output_dos_files: bool = True  # Whether to output DOS to dedicated files
    # energy_range: List[float] = field(default_factory=lambda: [-10.0, 10.0])  #
    # Energy range for DOS calculation (in eV)
    # smearing_width: float = 0.1  # Smearing width for DOS calculation (in eV)

    # bands_fdf_arguments: Dict[float,Any]= field(default_factory=dict)

    calculate_total_dos: bool = field(
        default=True,
        metadata={
            "description": (
                "A wrapper-level flag to enable the calculation of the total Density "
                "of States (DOS). In SIESTA, this is triggered by including a "
                "'ProjectedDensityOfStates' block."
            ),
            "SIESTA keyword": "%block ProjectedDensityOfStates",
        },
    )

    calculate_partial_dos: bool = field(
        default=False,
        metadata={
            "description": (
                "A wrapper-level flag to enable the calculation of the Partial (or "
                "Projected) Density of States (PDOS), which requires specifying atoms "
                "and orbitals in the 'ProjectedDensityOfStates' block."
            ),
            "SIESTA keyword": "%block ProjectedDensityOfStates",
        },
    )

    projected_dos_atoms: list[int] = field(
        default_factory=list,
        metadata={
            "description": (
                "A list of atom indices to be used for the Projected Density of States "
                "calculation. This list is used by the wrapper to generate the '%block "
                "ProjectedDensityOfStates'."
            ),
            "SIESTA keyword": "%block ProjectedDensityOfStates",
        },
    )

    band_structure_kpoints: list[list[float]] = field(
        default_factory=list,
        metadata={
            "description": (
                "A list of k-points defining the path for a band structure "
                "calculation. This list is used by the wrapper to generate the '%block "
                "BandLines'."
            ),
            "SIESTA keyword": "%block BandLines",
        },
    )

    output_dos_files: bool = field(
        default=True,
        metadata={
            "description": (
                "A wrapper-level flag to control whether the output DOS/PDOS files are "
                "generated. File writing in SIESTA is implicitly controlled by other "
                "settings."
            ),
            "SIESTA keyword": None,
        },
    )

    energy_range: list[float] = field(
        default_factory=lambda: [-10.0, 10.0],
        metadata={
            "description": (
                "The energy range [Emin, Emax] (in eV) for the DOS/PDOS calculation, "
                "which is specified within the 'ProjectedDensityOfStates' block."
            ),
            "SIESTA keyword": "%block ProjectedDensityOfStates",
        },
    )

    smearing_width: float = field(
        default=0.1,
        metadata={
            "description": (
                "The Gaussian or Lorentzian smearing width (in eV) applied to the "
                "DOS/PDOS plot, specified within the 'ProjectedDensityOfStates' block."
            ),
            "SIESTA keyword": "%block ProjectedDensityOfStates",
        },
    )

    n_energy_points: int | None = field(
        default=None,
        metadata={
            "description": (
                "Number of energy points for DOS/PDOS calculation. If None, calculated "
                "from energy_range and smearing_width."
            ),
            "SIESTA keyword": "%block ProjectedDensityOfStates",
        },
    )

    # bands_fdf_arguments: Dict[float, Any] = field(
    bands_fdf_arguments: OrderedDict[str, Any] = field(
        default_factory=OrderedDict,
        metadata={
            "description": (
                "A dictionary for any additional or arbitrary FDF flags related to "
                "bands or DOS calculations."
            ),
            "SIESTA keyword": None,
        },
    )

    comments: str = field(
        default="DensityOfStatesAndBandStructure Settings",
        metadata={
            "description": "Comment header for this dataclass section in the FDF file.",
            "SIESTA keyword": None,
        },
    )

    _registered: ClassVar[bool]

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                # Band structure
                "BandLinesScale",
                "%block BandLines",
                "%block BandPoints",
                "WriteKbands",
                "WriteBands",
                # Wavefunctions for bands
                "WFS.Write.For.Bands",
                "WFS.Band.Min",
                "WFS.Band.Max",
                "WFS.Energy.Min",
                "WFS.Energy.Max",
                # Selected wavefunctions
                "WaveFuncKPointsScale",
                "%block WaveFuncKPoints",
                "WriteWaveFunctions",
                # DOS k-grids
                "%block DOS.kgrid.MonkhorstPack",
                "DOS.kgrid.Cutoff",
                "DOS.kgrid.File",
                # PDOS (Projected DOS)
                "%block PDOS.kgrid.MonkhorstPack",
                "PDOS.kgrid.Cutoff",
                "PDOS.kgrid.File",
                "%block ProjectedDensityOfStates",
                # LDOS (Local DOS)
                "%block LDOS.kgrid.MonkhorstPack",
                "LDOS.kgrid.Cutoff",
                "LDOS.kgrid.File",
                "%block LocalDensityOfStates",
                # Population analysis
                "WriteMullikenPop",
                "Write.HirshfeldPop",
                "Write.VoronoiPop",
                "MullikenInSCF",
                "SpinInSCF",
                "PartialChargesAtEveryGeometry",
                "PartialChargesAtEverySCFStep",
                "COOP.Write",
            )
            self.__class__._registered = True  # noqa: SLF001 own-class registration guard

    def validate(self) -> None:
        """Validate the DOS and band-structure analysis options."""
        logger.info("DensityOfStatesAndBandStructure.validate()")
        if self.calculate_partial_dos and not self.projected_dos_atoms:
            raise ValueError(
                "Atom indices must be specified for partial DOS calculation."
            )
        if len(self.energy_range) != 2 or self.energy_range[0] >= self.energy_range[1]:
            raise ValueError(
                "Energy range must be a list of two values [min, max] with min < max."
            )
        print(  # noqa: T201 intentional validation diagnostic
            f"Validated: {self.calculate_total_dos=}, "
            f"{self.calculate_partial_dos=}, {self.projected_dos_atoms=}, "
            f"{self.band_structure_kpoints=}, {self.output_dos_files=}, "
            f"{self.energy_range=}, {self.smearing_width=}"
        )

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)

        Note:
            Simplified implementation focusing on commonly used parameters.
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            # Band structure
            if key_lower in ["bandlinesscale", "band_line_scale"]:
                self.band_line_scale = str(value)
            elif key_lower in ["writekbands", "write_k_bands"]:
                self.write_k_bands = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["writebands", "write_bands"]:
                self.write_bands = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            # BandLines block
            elif key_lower in ["%block bandlines", "bandlines"]:
                self.band_lines_block = value
            # Wavefunctions
            elif key_lower in ["writewavefunctions", "write_wave_functions"]:
                self.write_wave_functions = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            # ProjectedDensityOfStates block - store for direct output
            elif key_lower in [
                "%block projecteddensityofstates",
                "projecteddensityofstates",
            ]:
                self.projected_density_of_states_block = value
            # DOS k-grid blocks
            elif key_lower in [
                "%block dos.kgrid.monkhorstpack",
                "dos.kgrid.monkhorstpack",
            ]:
                self.dos_kgrid_monkhorst_pack_block = value
            elif key_lower in [
                "%block pdos.kgrid.monkhorstpack",
                "pdos.kgrid.monkhorstpack",
            ]:
                self.pdos_kgrid_monkhorst_pack_block = value
            elif key_lower in [
                "%block ldos.kgrid.monkhorstpack",
                "ldos.kgrid.monkhorstpack",
            ]:
                self.ldos_kgrid_monkhorst_pack_block = value
            # LocalDensityOfStates block
            elif key_lower in [
                "%block localdensityofstates",
                "localdensityofstates",
            ]:
                self.local_density_of_states_block = value

        # If user provided ProjectedDensityOfStates block, generate DOS block
        # This ensures the block is output for both StaticMaker and PDOSMaker
        if self.projected_density_of_states_block:
            self.calculate_total_dos = True
            self.calculate_partial_dos = True
            self.generate_dos_block()

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters

        Note:
            Only outputs band structure parameters (BandLinesScale, WriteKbands, etc.)
            when band structure is explicitly requested (band_lines_block is non-empty).
            DOS-only calculations don't need these parameters.
        """
        # Start with existing bands_fdf_arguments if populated (by generate_dos_block)
        # This preserves DOS block if it was already generated
        fdf: dict[str, Any]
        if hasattr(self, "bands_fdf_arguments") and self.bands_fdf_arguments:
            fdf = OrderedDict(self.bands_fdf_arguments)
        else:
            fdf = OrderedDict()
            fdf["#DensityOfStatesAndBandStructure"] = (
                "DensityOfStatesAndBandStructure Settings"
            )

        # Determine if band structure is being calculated
        # (indicated by non-empty band_lines_block or explicit band structure settings)
        doing_band_structure = (
            bool(self.band_lines_block) or self.write_bands or self.write_k_bands
        )

        # Only output band structure parameters if actually doing band structure
        if doing_band_structure:
            # BandLinesScale - write with default marker
            if self.band_line_scale == "ReciprocalLatticeVectors":
                fdf["BandLinesScale"] = (
                    "ReciprocalLatticeVectors  # SIESTA DEFAULT VALUE"
                )
            else:
                fdf["BandLinesScale"] = self.band_line_scale

            # WriteKbands - write with default marker
            if not self.write_k_bands:
                fdf["WriteKbands"] = "false  # SIESTA DEFAULT VALUE"
            else:
                fdf["WriteKbands"] = "true"

            # WriteBands - write with default marker
            if not self.write_bands:
                fdf["WriteBands"] = "false  # SIESTA DEFAULT VALUE"
            else:
                fdf["WriteBands"] = "true"

            # WriteWaveFunctions - write with default marker
            if not self.write_wave_functions:
                fdf["WriteWaveFunctions"] = "false  # SIESTA DEFAULT VALUE"
            else:
                fdf["WriteWaveFunctions"] = "true"

            # Band lines block - write if provided (no default marker, it's a block)
            if self.band_lines_block:
                fdf["%block BandLines"] = self.band_lines_block

        # DOS k-grid block - write if provided
        if self.dos_kgrid_monkhorst_pack_block:
            fdf["%block DOS.kgrid.MonkhorstPack"] = self.dos_kgrid_monkhorst_pack_block

        # PDOS k-grid block - write if provided
        if self.pdos_kgrid_monkhorst_pack_block:
            fdf["%block PDOS.kgrid.MonkhorstPack"] = (
                self.pdos_kgrid_monkhorst_pack_block
            )

        # LDOS k-grid block - write if provided
        if self.ldos_kgrid_monkhorst_pack_block:
            fdf["%block LDOS.kgrid.MonkhorstPack"] = (
                self.ldos_kgrid_monkhorst_pack_block
            )

        # NOTE: ProjectedDensityOfStates block is NOT output here.
        # For DOSMaker/PDOSMaker: handled by generate_dos_block() which is called in
        # get_parameter_updates()
        # For StaticMaker with user-provided block: the user's block is stored in
        # projected_density_of_states_block
        #   and generate_dos_block() is called automatically when dos_bands module is
        # activated
        # This avoids duplicates - there's only ONE place that outputs the block

        # LocalDensityOfStates block - write if user provided
        if self.local_density_of_states_block:
            if isinstance(self.local_density_of_states_block, list):
                fdf["%block LocalDensityOfStates"] = self.local_density_of_states_block
            elif isinstance(self.local_density_of_states_block, dict):
                # Convert dict to list format if needed
                fdf["%block LocalDensityOfStates"] = list(
                    self.local_density_of_states_block.values()
                )
            else:
                fdf["%block LocalDensityOfStates"] = [
                    self.local_density_of_states_block
                ]

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have DOS/band structure parameters
        # These are SIESTA-specific post-processing options
        return {}

    def generate_dos_block(self) -> None:
        """
        Generate the DOS calculation options block for the FDF file in SIESTA format.

        This method calls generate_fdf() first to ensure all base DOS/bands parameters
        are included with proper "# SIESTA DEFAULT VALUE" markers, then adds
        DOS-specific ProjectedDensityOfStates block.

        Creates %block ProjectedDensityOfStates which generates:
        - siesta.DOS (total density of states)
        - siesta.PDOS (projected DOS if atoms specified)
        - siesta.PDOS.xml (XML format)

        Format: EF Emin Emax dE nE units
        """
        logger.info("DensityOfStatesAndBandStructure.generate_dos_block()")

        if not (self.calculate_total_dos or self.calculate_partial_dos):
            return

        # Initialize DOS-only parameters (NOT band structure parameters like
        # BandLinesScale)
        # Band structure params are only added by generate_band_structure_block()
        self.bands_fdf_arguments = OrderedDict()

        # Add comment header
        self.bands_fdf_arguments["#DensityOfStatesAndBandStructure"] = (
            "DensityOfStatesAndBandStructure Settings"
        )

        # Add DOS/PDOS/LDOS k-grid blocks if provided
        if self.dos_kgrid_monkhorst_pack_block:
            self.bands_fdf_arguments["%block DOS.kgrid.MonkhorstPack"] = (
                self.dos_kgrid_monkhorst_pack_block
            )
        if self.pdos_kgrid_monkhorst_pack_block:
            self.bands_fdf_arguments["%block PDOS.kgrid.MonkhorstPack"] = (
                self.pdos_kgrid_monkhorst_pack_block
            )
        if self.ldos_kgrid_monkhorst_pack_block:
            self.bands_fdf_arguments["%block LDOS.kgrid.MonkhorstPack"] = (
                self.ldos_kgrid_monkhorst_pack_block
            )

        # Check if user already provided a ProjectedDensityOfStates block directly
        # If so, use it instead of generating a new one
        if self.projected_density_of_states_block:
            # User provided their own block - use it directly
            if isinstance(self.projected_density_of_states_block, list):
                self.bands_fdf_arguments["ProjectedDensityOfStates"] = (
                    self.projected_density_of_states_block
                )
            else:
                self.bands_fdf_arguments["ProjectedDensityOfStates"] = [
                    self.projected_density_of_states_block
                ]
            # Clear the block so generate_fdf() won't output it again (avoid duplicates)
            self.projected_density_of_states_block = {}
            logger.info("Using user-provided ProjectedDensityOfStates block")
            return

        # Get energy range and smearing parameters
        emin, emax = self.energy_range
        de = self.smearing_width

        # Use user-provided n_energy_points if available, otherwise calculate
        if self.n_energy_points is not None:
            ne = self.n_energy_points
        else:
            ne = int((emax - emin) / de) + 1

        # Build ProjectedDensityOfStates block
        pdos_lines = []

        # First line: energy grid (EF means relative to Fermi level)
        pdos_lines.append(f"EF {emin:.3f} {emax:.3f} {de:.3f} {ne} eV")

        # SIESTA PDOS format:
        # - Just energy line: Total DOS only
        # - No additional lines needed for PDOS - SIESTA generates PDOS automatically
        #   when ProjectedDensityOfStates block is present
        # Note: projected_dos_atoms is not used in the FDF block itself
        # SIESTA will generate PDOS for all atoms when this block is present

        # Store in bands_fdf_arguments (will be merged into main fdf_arguments)
        self.bands_fdf_arguments["ProjectedDensityOfStates"] = pdos_lines

        logger.info(f"Generated DOS block with {len(pdos_lines)} lines")

    def generate_band_structure_block(self, structure: "Structure") -> None:
        """
        Generate the band structure calculation options block for the FDF file.

        This method calls generate_fdf() first to ensure all base DOS/bands parameters
        are included with proper "# SIESTA DEFAULT VALUE" markers, then adds
        band-specific parameters.
        """
        logger.info("DensityOfStatesAndBandStructure.generate_band_structure_block()")

        # Enable band structure output so generate_fdf() includes band params
        self.write_bands = True

        # Start with base parameters from generate_fdf() (includes default markers)
        self.bands_fdf_arguments = cast("OrderedDict[str, Any]", self.generate_fdf())

        # Add band-specific header and parameters
        temp = OrderedDict()
        temp["#BandStructure"] = "BandStructure"
        temp.update(self.bands_fdf_arguments)
        temp["WaveFuncKPointsScale"] = self.wave_func_k_point_scale

        # Add BandLines block
        self.wave_func_k_points_block = {
            "BandLines": band_paymatgen_to_siesta(
                structure=structure, interpolations=[50]
            )
        }
        temp.update(self.wave_func_k_points_block)

        self.bands_fdf_arguments = temp

    @classmethod
    def setup_dos_bands_settings(
        cls,
        user_params: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG003 kept for interface compatibility
    ) -> "DensityOfStatesAndBandStructure":
        """
        Create and configure a DensityOfStatesAndBandStructure from user parameters.

        This method handles proper key normalization, type conversion, and fuzzy
        matching to configure DOS and band structure settings from user parameters.

        Args:
            user_params: Dictionary of user-defined parameters (case-insensitive,
                        may include dots).
                        If None or empty, all default values are used.
            **kwargs: Additional keyword arguments to override or supplement
                user_params.

        Returns
        -------
            DensityOfStatesAndBandStructure: Configured instance with all fields set.
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]DensityOfStatesAndBandStructure.setup_dos_bands_settings()[/green]"
            )

        # Initialize instance with defaults
        instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default "
                    "DOS/Bands values.[/blue]"
                )
            return instance

        # Get valid attribute names (lowercase for comparison)
        dos_attributes = {
            field.name.lower()
            for field in fields(cls)
            if not field.name.startswith("_") and field.name != "CONSOLE_VERBOSITY"
        }
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                f"[blue]Available DensityOfStatesAndBandStructure "
                f"attributes: {dos_attributes}[/blue]"
            )

        # Process user parameters
        import re

        for key, value in user_params.items():
            # Normalize key: handle camelCase properly
            key_with_underscores = re.sub(r"([a-z])([A-Z])", r"\1_\2", key)
            key_normalized = key_with_underscores.replace(".", "_").lower()

            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Processing key: {key} -> {key_normalized}, "
                    f"value: {value}[/blue]"
                )

            # Check if normalized key matches any attribute
            matched_attr = None
            if key_normalized in dos_attributes:
                matched_attr = key_normalized
            else:
                # Fuzzy match: remove all underscores and compare
                key_no_underscores = key_normalized.replace("_", "")
                for attr in dos_attributes:
                    if attr.replace("_", "") == key_no_underscores:
                        matched_attr = attr
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                            console.print(
                                f"[blue]Fuzzy matched: {key_normalized} -> "
                                f"{attr}[/blue]"
                            )
                        break

            if matched_attr:
                # Find the original attribute name (preserving case)
                original_key = next(
                    field.name
                    for field in fields(cls)
                    if field.name.lower() == matched_attr
                )

                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                    console.print(
                        f"[blue]Matched field: {original_key} = {value}[/blue]"
                    )

                # Handle type conversion for specific field types
                # Dict/OrderedDict fields (blocks and fdf_arguments)
                if original_key in [
                    "band_lines_block",
                    "band_points_block",
                    "wave_func_k_points_block",
                    "dos_kgrid_monkhorst_pack_block",
                    "pdos_kgrid_monkhorst_pack_block",
                    "ldos_kgrid_monkhorst_pack_block",
                    "projected_density_of_states_block",
                    "local_density_of_states_block",
                    "bands_fdf_arguments",
                ]:
                    if isinstance(value, (dict, OrderedDict)):
                        setattr(instance, original_key, value)
                    elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            f"[yellow]Invalid type for {original_key}: "
                            f"expected dict, got {type(value)}[/yellow]"
                        )

                # Boolean fields
                elif original_key in [
                    "write_k_bands",
                    "write_bands",
                    "wfs_write_for_bands",
                    "write_wave_functions",
                    "mulliken_in_scf",
                    "spin_in_scf",
                    "write_hirshfeld_pop",
                    "write_voronoi_pop",
                    "partial_charges_at_every_geometry",
                    "partial_charges_at_every_scf_step",
                    "coop_write",
                    "calculate_total_dos",
                    "calculate_partial_dos",
                    "output_dos_files",
                ]:
                    bool_value = value
                    if isinstance(value, str):
                        bool_value = value.lower() in ("true", "t", "1", "yes")
                    setattr(instance, original_key, bool(bool_value))

                # Integer fields
                elif original_key in [
                    "wfs_band_min",
                    "wfs_band_max",
                    "write_mullkin_pop",
                ]:
                    try:
                        if value is not None:
                            setattr(instance, original_key, int(value))
                        else:
                            setattr(instance, original_key, None)
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[yellow]Could not convert "
                                f"{original_key}={value} to int[/yellow]"
                            )

                # Float fields (including optional)
                elif original_key in [
                    "dos_kgrid_cutoff",
                    "pdos_kgrid_cutoff",
                    "ldos_kgrid_cutoff",
                    "wfs_energy_min",
                    "wfs_energy_max",
                    "smearing_width",
                ]:
                    try:
                        if value is not None:
                            setattr(instance, original_key, float(value))
                        else:
                            setattr(instance, original_key, None)
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[yellow]Could not convert "
                                f"{original_key}={value} to float[/yellow]"
                            )

                # List fields
                elif original_key in [
                    "projected_dos_atoms",
                    "band_structure_kpoints",
                    "energy_range",
                ]:
                    if isinstance(value, list):
                        setattr(instance, original_key, value)
                    elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            f"[yellow]Invalid type for {original_key}: "
                            f"expected list, got {type(value)}[/yellow]"
                        )

                # String fields
                elif original_key in [
                    "band_line_scale",
                    "wave_func_k_point_scale",
                    "dos_kgrid_file",
                    "pdos_kgrid_file",
                    "ldos_kgrid_file",
                ]:
                    setattr(
                        instance,
                        original_key,
                        str(value) if value is not None else None,
                    )

                # Default: direct assignment
                else:
                    setattr(instance, original_key, value)

            elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                console.print(
                    f"[yellow]Unrecognized parameter: {key} "
                    f"(normalized: {key_normalized})[/yellow]"
                )

        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(
                "[green]DensityOfStatesAndBandStructure instance "
                "configured successfully.[/green]"
            )

        return instance
