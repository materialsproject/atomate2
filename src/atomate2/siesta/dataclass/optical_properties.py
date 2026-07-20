"""
Module defining base SIESTA input set and generator for OpticalProperties.

class OpticalProperties

Based on User's Guide Siesta 5.4.0
Section:  6.20 Optical properties
"""

# Metadata

__all__ = ["OpticalProperties"]

import logging
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@dataclass
class OpticalProperties(FDFDataclass):
    """Data class to analysis optical properties options for SIESTA input."""

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.ERROR
    )  # Default to show info & errors messages
    # ------------------------
    # 6.20 Optical properties
    # ------------------------
    # optical_calculation: bool = False # OpticalCalculation false
    # optical_energy_minimum: float = -10.0 # Optical.Energy.Minimum 0 Ry
    # optical_energy_maximum: float =  20.0 # Optical.Energy.Maximum 10 Ry
    # optical_broaden: float = 0.1 # Optical.Broaden 0 Ry
    # optical_scissor: float = 0.0 # Optical.Scissor 0 Ry
    # optical_number_of_bands: int = None  # Optical.NumberOfBands all bands
    # optical_mesh_block: List[str] = field(default_factory=lambda:['10 10 10']) # ['10 10 10'] #Dict[float,Any]= field(default_factory=dict) # %block Optical.Mesh 〈None〉  # noqa: E501
    # optical_offset_mesh: bool = False # Optical.OffsetMesh false
    # optical_polarization_type: str = 'polycrystal' # Optical.PolarizationType polycrystal  # noqa: E501
    # optical_vector_block: Dict[float,Any]= field(default_factory=dict) # %block Optical.Vector 〈None〉  # noqa: E501
    optical_calculation: bool = field(
        default=False,
        metadata={
            "description": (
                "A wrapper-level flag to enable the calculation of optical properties. "
                "This will activate the relevant 'Optical' keywords in the input."
            ),
            "SIESTA keyword": "OpticalCalculation",
        },
    )

    optical_energy_minimum: float = field(
        default=-10.0,
        metadata={
            "description": (
                "The minimum energy (in Rydberg) of the photon range for which the "
                "optical spectrum is calculated."
            ),
            "SIESTA keyword": "Optical.Energy.Minimum",
        },
    )

    optical_energy_maximum: float = field(
        default=20.0,
        metadata={
            "description": (
                "The maximum energy (in Rydberg) of the photon range for which the "
                "optical spectrum is calculated."
            ),
            "SIESTA keyword": "Optical.Energy.Maximum",
        },
    )

    optical_broaden: float = field(
        default=0.1,
        metadata={
            "description": (
                "An energy broadening (in Rydberg) applied to the calculated optical "
                "spectrum to aid visualization."
            ),
            "SIESTA keyword": "Optical.Broaden",
        },
    )

    optical_scissor: float = field(
        default=0.0,
        metadata={
            "description": (
                "A rigid energy shift (in Rydberg), known as a 'scissor operator', "
                "applied to the conduction bands to correct for band-gap errors."
            ),
            "SIESTA keyword": "Optical.Scissor",
        },
    )

    optical_number_of_bands: int = field(
        default=None,
        metadata={
            "description": (
                "The number of bands to be included in the optical properties "
                "calculation. Defaults to all available bands if not set."
            ),
            "SIESTA keyword": "Optical.NumberOfBands",
        },
    )

    optical_mesh_block: list[str] = field(
        default_factory=lambda: ["10 10 10"],
        metadata={
            "description": (
                "A block to define the dimensions of a specific Monkhorst-Pack k-point "
                "grid for the optical calculation, which is typically denser than the "
                "SCF grid."
            ),
            "SIESTA keyword": "%block Optical.Mesh",
        },
    )

    optical_offset_mesh: bool = field(
        default=False,
        metadata={
            "description": (
                "A flag to control whether to use an offset for the k-point grid in "
                "the optical calculation."
            ),
            "SIESTA keyword": "Optical.OffsetMesh",
        },
    )

    optical_polarization_type: str = field(
        default="polycrystal",
        metadata={
            "description": (
                "Specifies the type of light polarization to be considered. Options "
                "are 'polarized', 'unpolarized', or 'polycrystal'."
            ),
            "SIESTA keyword": "Optical.PolarizationType",
        },
    )

    optical_vector_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to specify the electric field polarization vector when "
                "'Optical.PolarizationType' is set to 'polarized'."
            ),
            "SIESTA keyword": "%block Optical.Vector",
        },
    )

    optical_number_of_energies: int | None = field(
        default=None,
        metadata={
            "description": (
                "The number of energy points for the optical spectrum. If not set, "
                "SIESTA uses an internally determined value."
            ),
            "SIESTA keyword": "Optical.NumberOfEnergies",
        },
    )

    optical_calculation_type: str | None = field(
        default=None,
        metadata={
            "description": (
                "The type of optical calculation (e.g., 'absorption', 'optical')."
            ),
            "SIESTA keyword": "Optical.CalculationType",
        },
    )
    # ------------------------------
    # 6.21 Macroscopic polarization
    # ------------------------------
    # polarization_grids_block: Dict[float,Any]= field(default_factory=dict) # %block PolarizationGrids 〈None〉  # noqa: E501
    # born_charge: bool = False # BornCharge false
    # calculate_mulliken_charges: bool = True  # Flag to indicate if Mulliken charges should be calculated  # noqa: E501
    # calculate_overlap_populations: bool = False  # Flag to indicate if overlap populations should be calculated  # noqa: E501
    # calculate_coop_cohp: bool = False  # Flag to indicate if COOP/COHP analysis should be performed  # noqa: E501
    # optical_properties: bool = False  # Flag to indicate if optical properties should be calculated  # noqa: E501
    # optical_energy_range: List[float] = field(default_factory=lambda: [0.0, 10.0])  # Energy range for optical calculations (in eV)  # noqa: E501
    # #optical_k_points: List[List[float]] = field(default_factory=list)  # List of k-points for optical properties calculation  # noqa: E501
    # polarization_directions: List[List[float]] = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Polarization directions  # noqa: E501
    # optical_fdf_arguments: Dict[float,Any]= field(default_factory=dict)
    polarization_grids_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to define fine-grained grids for the calculation of electric "
                "polarization using the Berry phase method."
            ),
            "SIESTA keyword": "%block PolarizationGrids",
        },
    )

    born_charge: bool = field(
        default=False,
        metadata={
            "description": (
                "A flag to enable the calculation of Born effective charges."
            ),
            "SIESTA keyword": "BornCharge",
        },
    )

    calculate_mulliken_charges: bool = field(
        default=True,
        metadata={
            "description": (
                "A wrapper-level flag to enable Mulliken population analysis. Sets "
                "'WriteMullikenPop' to 1 or higher."
            ),
            "SIESTA keyword": "WriteMullikenPop",
        },
    )

    calculate_overlap_populations: bool = field(
        default=False,
        metadata={
            "description": (
                "A wrapper-level flag to enable Mulliken overlap population analysis. "
                "Sets 'WriteMullikenPop' to 3."
            ),
            "SIESTA keyword": "WriteMullikenPop",
        },
    )

    calculate_coop_cohp: bool = field(
        default=False,
        metadata={
            "description": (
                "A wrapper-level flag to enable COOP/COHP analysis for chemical "
                "bonding. Sets 'COOP.Write' to true."
            ),
            "SIESTA keyword": "COOP.Write",
        },
    )

    optical_properties: bool = field(
        default=False,
        metadata={
            "description": (
                "A wrapper-level flag to enable the calculation of optical properties. "
                "Sets 'OpticalCalculation' to true."
            ),
            "SIESTA keyword": "OpticalCalculation",
        },
    )

    optical_energy_range: list[float] = field(
        default_factory=lambda: [0.0, 10.0],
        metadata={
            "description": (
                "The energy range [Emin, Emax] for optical calculations. Corresponds "
                "to 'Optical.Energy.Minimum' and 'Optical.Energy.Maximum'."
            ),
            "SIESTA keyword": "Optical.Energy.Minimum, Optical.Energy.Maximum",
        },
    )

    polarization_directions: list[list[float]] = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        metadata={
            "description": (
                "Defines the electric field polarization vectors for the optical "
                "calculation. This is used to generate the '%block Optical.Vector'."
            ),
            "SIESTA keyword": "%block Optical.Vector",
        },
    )

    comments: str = field(
        default="OpticalProperties Settings",
        metadata={
            "description": "Comment header for this dataclass section in the FDF file.",
            "SIESTA keyword": None,
        },
    )

    optical_fdf_arguments: OrderedDict[str, Any] = field(
        default_factory=OrderedDict,
        metadata={
            "description": (
                "A dictionary for any additional or arbitrary FDF flags related to "
                "optical properties."
            ),
            "SIESTA keyword": None,
        },
    )

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "OpticalCalculation",
                "Optical.Energy.Minimum",
                "Optical.Energy.Maximum",
                "Optical.Broaden",
                "Optical.Scissor",
                "Optical.NumberOfBands",
                "Optical.NumberOfEnergies",
                "Optical.CalculationType",
                "%block Optical.Mesh",
                "Optical.OffsetMesh",
                "Optical.PolarizationType",
                "%block Optical.Vector",
            )
            self.__class__._registered = True  # noqa: SLF001 own class-level guard

    def validate(self) -> None:
        """Validate the chemical analysis and optical properties options."""
        logger.info("OpticalProperties.validate()")
        allowed_optical_polarization_type = ["polycrystal", "polarized", "unpolarized"]
        if self.optical_calculation:
            if self.optical_polarization_type not in allowed_optical_polarization_type:
                raise ValueError(
                    f"Invalid optical_polarization_type "
                    f"'{self.optical_polarization_type}'. Allowed values are: "
                    f"{allowed_optical_polarization_type}"
                )
            # if len(self.optical_energy_range) != 2 or self.optical_energy_range[0] >= self.optical_energy_range[1]:  # noqa: E501
            #    raise ValueError("Energy range for optical properties must be a list of two values [min, max] with min < max.")  # noqa: E501
            if not self.optical_mesh_block:
                raise ValueError(
                    " optical_mesh_block must be specified for optical properties "
                    "calculation."
                )
        # print(f"Validation: OpticalProperties DONE!")
        # print(f"Validated: {self.calculate_mulliken_charges=}, {self.calculate_overlap_populations=}, {self.calculate_coop_cohp=}, {self.optical_properties=}, {self.optical_energy_range=}, {self.optical_k_points=}, {self.polarization_directions=}")  # noqa: E501

        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(
                "[green]Validation & Generation: "
                "[yellow]OpticalProperties[/yellow] Successful![/green]"
            )

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)

        Note:
            Simplified implementation for common optical parameters.
        """
        from atomate2.siesta.dataclass.units import parse_energy

        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in ["opticalcalculation", "optical_calculation"]:
                self.optical_calculation = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["optical.energy.minimum", "optical_energy_minimum"]:
                self.optical_energy_minimum = parse_energy(value, target_unit="Ry")
            elif key_lower in ["optical.energy.maximum", "optical_energy_maximum"]:
                self.optical_energy_maximum = parse_energy(value, target_unit="Ry")
            elif key_lower in ["optical.broaden", "optical_broaden"]:
                self.optical_broaden = parse_energy(value, target_unit="Ry")
            elif key_lower in ["optical.scissor", "optical_scissor"]:
                self.optical_scissor = parse_energy(value, target_unit="Ry")
            elif key_lower in ["optical.numberofbands", "optical_number_of_bands"]:
                self.optical_number_of_bands = int(value)
            elif key_lower in [
                "optical.numberofenergies",
                "optical_number_of_energies",
            ]:
                self.optical_number_of_energies = int(value)
            elif key_lower in ["optical.calculationtype", "optical_calculation_type"]:
                self.optical_calculation_type = str(value)

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters
        """
        fdf: dict[str, Any] = OrderedDict()

        # Add comment header
        fdf["#OpticalProperties"] = "OpticalProperties Settings"

        # OpticalCalculation - always write with default marker
        if not self.optical_calculation:
            fdf["OpticalCalculation"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["OpticalCalculation"] = "true"

        # Only write optical parameters if optical calculation is enabled
        if self.optical_calculation:
            # Optical.Energy.Minimum - always write with default marker
            if self.optical_energy_minimum == -10.0:
                fdf["Optical.Energy.Minimum"] = "-10.0 Ry  # SIESTA DEFAULT VALUE"
            else:
                fdf["Optical.Energy.Minimum"] = f"{self.optical_energy_minimum} Ry"

            # Optical.Energy.Maximum - always write with default marker
            if self.optical_energy_maximum == 20.0:
                fdf["Optical.Energy.Maximum"] = "20.0 Ry  # SIESTA DEFAULT VALUE"
            else:
                fdf["Optical.Energy.Maximum"] = f"{self.optical_energy_maximum} Ry"

            # Optical.Broaden - always write with default marker
            if self.optical_broaden == 0.1:
                fdf["Optical.Broaden"] = "0.1 Ry  # SIESTA DEFAULT VALUE"
            else:
                fdf["Optical.Broaden"] = f"{self.optical_broaden} Ry"

            # Optical.Scissor - always write with default marker
            if self.optical_scissor == 0.0:
                fdf["Optical.Scissor"] = "0.0 Ry  # SIESTA DEFAULT VALUE"
            else:
                fdf["Optical.Scissor"] = f"{self.optical_scissor} Ry"

            # Optical.NumberOfBands - write if set (no default, optional)
            if self.optical_number_of_bands:
                fdf["Optical.NumberOfBands"] = str(self.optical_number_of_bands)

            # Optical.NumberOfEnergies - write if set (no default, optional)
            if self.optical_number_of_energies:
                fdf["Optical.NumberOfEnergies"] = str(self.optical_number_of_energies)

            # Optical.CalculationType - write if set (no default, optional)
            if self.optical_calculation_type:
                fdf["Optical.CalculationType"] = self.optical_calculation_type

            # Optical.PolarizationType - always write with default marker
            if self.optical_polarization_type == "polycrystal":
                fdf["Optical.PolarizationType"] = "polycrystal  # SIESTA DEFAULT VALUE"
            else:
                fdf["Optical.PolarizationType"] = self.optical_polarization_type

            # %block Optical.Mesh - write if provided
            if self.optical_mesh_block:
                fdf["%block Optical.Mesh"] = self.optical_mesh_block

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have optical properties parameters
        # These are SIESTA-specific post-processing options
        return {}

    def generate_optical_properties_block(self) -> None:
        """Generate the optical properties options block for the FDF file."""
        logger.info("OpticalProperties.generate_optical_properties_block()")
        if self.optical_calculation:
            # Add comment header
            if self.comments:
                self.optical_fdf_arguments["#OpticalProperties"] = self.comments

            self.optical_fdf_arguments.update(
                OrderedDict(
                    [
                        ("OpticalCalculation", f"{self.optical_calculation}"),
                        ("Optical.Energy.Minimum", f"{self.optical_energy_minimum} eV"),
                        ("Optical.Energy.Maximum", f"{self.optical_energy_maximum} eV"),
                        ("Optical.Broaden", f"{self.optical_broaden} eV"),
                        ("Optical.Scissor", f"{self.optical_scissor} eV"),
                        (
                            "Optical.PolarizationType",
                            f"{self.optical_polarization_type}",
                        ),
                    ]
                )
            )
            # self.optical_mesh_block = ['5 5 5']
            self.optical_mesh_block_ = {"Optical.Mesh": self.optical_mesh_block}

            self.optical_fdf_arguments.update(self.optical_mesh_block_)
        # self.bands_fdf_arguments.update(self.wave_func_k_points_block)

    @classmethod
    def setup_optical_settings(
        cls,
        user_params: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG003 optional interface passthrough
    ) -> "OpticalProperties":
        """
        Create and configure a OpticalProperties instance with full parameter parsing.

        This method handles proper key normalization, type conversion, and fuzzy
        matching to configure optical properties settings from user parameters.

        Args:
            user_params: Dictionary of user-defined parameters (case-insensitive,
                may include dots). If None or empty, all default values are used.
            **kwargs: Additional keyword arguments to override or supplement
                user_params.

        Returns
        -------
            OpticalProperties: Configured instance with all fields set.
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]OpticalProperties.setup_optical_settings()[/green]")

        # Initialize instance with defaults
        instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default Optical "
                    "values.[/blue]"
                )
            return instance

        # Get valid attribute names (lowercase for comparison)
        optical_attributes = {
            field.name.lower()
            for field in fields(cls)
            if not field.name.startswith("_") and field.name != "CONSOLE_VERBOSITY"
        }
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                f"[blue]Available OpticalProperties attributes: "
                f"{optical_attributes}[/blue]"
            )

        # Process user parameters
        import re

        for key, value in user_params.items():
            # Normalize key: handle camelCase properly
            key_with_underscores = re.sub(r"([a-z])([A-Z])", r"\1_\2", key)
            key_normalized = key_with_underscores.replace(".", "_").lower()

            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Processing key: {key} -> {key_normalized}, value: "
                    f"{value}[/blue]"
                )

            # Check if normalized key matches any attribute
            matched_attr = None
            if key_normalized in optical_attributes:
                matched_attr = key_normalized
            else:
                # Fuzzy match: remove all underscores and compare
                key_no_underscores = key_normalized.replace("_", "")
                for attr in optical_attributes:
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
                # Dict/OrderedDict fields
                if original_key in [
                    "optical_vector_block",
                    "polarization_grids_block",
                    "optical_fdf_arguments",
                ]:
                    if isinstance(value, (dict, OrderedDict)):
                        setattr(instance, original_key, value)
                    elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            f"[yellow]Invalid type for {original_key}: expected "
                            f"dict, got {type(value)}[/yellow]"
                        )

                # Boolean fields
                elif original_key in [
                    "optical_calculation",
                    "optical_offset_mesh",
                    "born_charge",
                    "calculate_mulliken_charges",
                    "calculate_overlap_populations",
                    "calculate_coop_cohp",
                    "optical_properties",
                ]:
                    bool_value = value
                    if isinstance(bool_value, str):
                        bool_value = bool_value.lower() in ("true", "t", "1", "yes")
                    setattr(instance, original_key, bool(bool_value))

                # Integer fields (optional)
                elif original_key == "optical_number_of_bands":
                    try:
                        if value is not None:
                            setattr(instance, original_key, int(value))
                        else:
                            setattr(instance, original_key, None)
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[yellow]Could not convert {original_key}={value} "
                                f"to int[/yellow]"
                            )

                # Float fields
                elif original_key in [
                    "optical_energy_minimum",
                    "optical_energy_maximum",
                    "optical_broaden",
                    "optical_scissor",
                ]:
                    try:
                        setattr(instance, original_key, float(value))
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[yellow]Could not convert {original_key}={value} "
                                f"to float[/yellow]"
                            )

                # List fields
                elif original_key in [
                    "optical_mesh_block",
                    "optical_energy_range",
                    "polarization_directions",
                ]:
                    if isinstance(value, list):
                        setattr(instance, original_key, value)
                    elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            f"[yellow]Invalid type for {original_key}: expected "
                            f"list, got {type(value)}[/yellow]"
                        )

                # String fields
                elif original_key == "optical_polarization_type":
                    setattr(instance, original_key, str(value))

                # Default: direct assignment
                else:
                    setattr(instance, original_key, value)

            elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                console.print(
                    f"[yellow]Unrecognized parameter: {key} (normalized: "
                    f"{key_normalized})[/yellow]"
                )

        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(
                "[green]OpticalProperties instance configured successfully.[/green]"
            )

        return instance
