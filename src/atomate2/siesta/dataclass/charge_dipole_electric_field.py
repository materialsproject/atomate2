"""
Data class to manage charge, dipole, and electric field options for SIESTA input.

class ChargeDipoleElectricField

Based on User's Guide Siesta 5.4.0
Section: 6.23 Systems with net charge or dipole, and electric fields
         6.23.1 Bulk current
"""

# Metadata

__all__ = ["ChargeDipoleElectricField"]

import logging
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@dataclass
class ChargeDipoleElectricField(FDFDataclass):
    """Manage charge, dipole, and electric field options for SIESTA input."""

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.ERROR
    )  # Default to show errors only

    # -----------------------------------------------------------
    # 6.23 Systems with net charge or dipole, and electric fields
    # -----------------------------------------------------------
    # net_charge: int = 0 # NetCharge 0
    # simulate_doping: bool = False  # SimulateDoping false
    # external_electric_field_block:  Dict[float,Any]= field(default_factory=dict)
    #   # %block ExternalElectricField 〈None〉
    # slab_dipole_correction: str = ''
    #   # Slab.DipoleCorrection ?|true|false|charge|vacuum|none
    # slab_dipole_correction_origin_block: Dict[float,Any]= field(default_factory=dict)
    #   # %block Slab.DipoleCorrection.Origin 〈None〉
    # slab_dipole_correction_vacuum_block: Dict[float,Any]= field(default_factory=dict)
    #   # %block Slab.DipoleCorrection.Vacuum 〈None〉
    # geometry_hartree_block: Dict[float,Any]= field(default_factory=dict)
    #   # %block Geometry.Hartree 〈None〉
    # geometry_charge_block: Dict[float,Any]= field(default_factory=dict)
    #   # %block Geometry.Charge 〈None〉
    net_charge: float = field(
        default=0.0,
        metadata={
            "description": (
                "Sets the total net charge of the system in units of electron "
                "charge. A compensating background charge is added "
                "automatically."
            ),
            "SIESTA keyword": "NetCharge",
        },
    )

    simulate_doping: bool = field(
        default=False,
        metadata={
            "description": (
                "A flag to enable a model for simulating doping by adding "
                "fractional charges to the nuclei, as an alternative to "
                "NetCharge."
            ),
            "SIESTA keyword": "SimulateDoping",
        },
    )

    external_electric_field_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to define a uniform external electric field applied "
                "to the system, specifying its direction and magnitude."
            ),
            "SIESTA keyword": "%block ExternalElectricField",
        },
    )

    slab_dipole_correction: str = field(
        default="",
        metadata={
            "description": (
                "Enables a dipole correction for slab geometries to cancel "
                "spurious interactions between periodic images. Options: "
                "'true', 'false', 'charge', 'vacuum', 'none'."
            ),
            "SIESTA keyword": "Slab.DipoleCorrection",
        },
    )

    slab_dipole_correction_origin_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to specify the origin point for the slab dipole "
                "correction potential."
            ),
            "SIESTA keyword": "%block Slab.DipoleCorrection.Origin",
        },
    )

    slab_dipole_correction_vacuum_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to specify the vacuum region of the cell for the "
                "slab dipole correction algorithm."
            ),
            "SIESTA keyword": "%block Slab.DipoleCorrection.Vacuum",
        },
    )

    geometry_hartree_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to define a custom analytical shape for the Hartree "
                "potential, useful for modeling electrostatic gates or "
                "environments."
            ),
            "SIESTA keyword": "%block Geometry.Hartree",
        },
    )

    geometry_charge_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to define a fictitious, continuous charge "
                "distribution (e.g., a charged plane) within the simulation "
                "cell."
            ),
            "SIESTA keyword": "%block Geometry.Charge",
        },
    )

    # -------------------
    # 6.23.1 Bulk current
    # -------------------
    # bulk_bias_voltage: float = 0.0 # BulkBias.Voltage 0. eV
    # bulk_bias_direction_block: Dict[float,Any]= field(default_factory=dict)
    #   # %block BulkBias.Direction 〈None〉
    # bulk_bias_tolerance: float = 1e-15 # BulkBias.Tolerance 10-15
    # bulk_bias_current: bool = True # BulkBias.Current true
    bulk_bias_voltage: float = field(
        default=0.0,
        metadata={
            "description": (
                "Sets the magnitude of the voltage bias (in eV) to be applied "
                "between the electrodes in a transport calculation."
            ),
            "SIESTA keyword": "BulkBias.Voltage",
            "unit": "eV",
        },
    )

    bulk_bias_direction_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to define the direction vector along which the bulk "
                "voltage bias is applied."
            ),
            "SIESTA keyword": "%block BulkBias.Direction",
        },
    )

    bulk_bias_tolerance: float = field(
        default=1e-15,
        metadata={
            "description": (
                "A convergence tolerance for the self-consistent calculation "
                "under the applied bulk bias."
            ),
            "SIESTA keyword": "BulkBias.Tolerance",
        },
    )

    bulk_bias_current: bool = field(
        default=True,
        metadata={
            "description": (
                "A flag to enable the calculation of the electrical current "
                "resulting from the applied bias."
            ),
            "SIESTA keyword": "BulkBias.Current",
        },
    )

    comments: str = field(
        default="ChargeDipoleElectricField Settings",
        metadata={
            "description": "Comment header for this dataclass section in the FDF file.",
            "SIESTA keyword": None,
        },
    )

    charge_dipole_fdf_arguments: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A dictionary for any additional or arbitrary FDF flags "
                "related to charge, dipole, and electric field."
            ),
            "SIESTA keyword": None,
        },
    )

    _registered: ClassVar[bool]

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "NetCharge",
                "SimulateDoping",
                "%block ExternalElectricField",
                "Slab.DipoleCorrection",
                "%block Slab.DipoleCorrection.Origin",
                "%block Slab.DipoleCorrection.Vacuum",
                "%block Geometry.Hartree",
                "%block Geometry.Charge",
                "BulkBias.Voltage",
                "%block BulkBias.Direction",
                "BulkBias.Tolerance",
                "BulkBias.Current",
            )
            self.__class__._registered = True  # noqa: SLF001 class-level flag

    def validate(self) -> None:
        """
        Validate charge, dipole, and electric field settings.

        Checks configuration for net charge, external electric fields, slab dipole
        corrections, and bulk bias parameters. Ensures settings are consistent for
        charged systems, doped systems, and transport calculations.

        Raises
        ------
        ValueError
            If charge/field parameters are invalid or slab dipole correction
            mode is not recognized
        """
        logger.info("ChargeDipoleElectricField.validate()")

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        from atomate2.siesta.dataclass.units import parse_energy

        for key, value in fdf_dict.items():
            key_lower = key.lower()

            # Net charge and doping
            if key_lower in ["netcharge", "net_charge"]:
                self.net_charge = float(value)
            elif key_lower in ["simulatedoping", "simulate_doping"]:
                self.simulate_doping = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )

            # Electric field and dipole correction
            elif key_lower in [
                "%block externalelectricfield",
                "external_electric_field_block",
            ]:
                self.external_electric_field_block = value
            elif key_lower in ["slab.dipolecorrection", "slab_dipole_correction"]:
                self.slab_dipole_correction = str(value)
            elif key_lower in [
                "%block slab.dipolecorrection.origin",
                "slab_dipole_correction_origin_block",
            ]:
                self.slab_dipole_correction_origin_block = value
            elif key_lower in [
                "%block slab.dipolecorrection.vacuum",
                "slab_dipole_correction_vacuum_block",
            ]:
                self.slab_dipole_correction_vacuum_block = value
            elif key_lower in ["%block geometry.hartree", "geometry_hartree_block"]:
                self.geometry_hartree_block = value
            elif key_lower in ["%block geometry.charge", "geometry_charge_block"]:
                self.geometry_charge_block = value

            # Bulk bias settings
            elif key_lower in ["bulkbias.voltage", "bulk_bias_voltage"]:
                # Parse energy with units
                self.bulk_bias_voltage = parse_energy(value, target_unit="eV")
            elif key_lower in [
                "%block bulkbias.direction",
                "bulk_bias_direction_block",
            ]:
                self.bulk_bias_direction_block = value
            elif key_lower in ["bulkbias.tolerance", "bulk_bias_tolerance"]:
                self.bulk_bias_tolerance = float(value)
            elif key_lower in ["bulkbias.current", "bulk_bias_current"]:
                self.bulk_bias_current = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters
        """
        fdf: dict[str, Any] = {}

        # Net charge and doping
        if self.net_charge != 0:
            fdf["NetCharge"] = str(self.net_charge)
        if self.simulate_doping:
            fdf["SimulateDoping"] = "true"

        # External electric field
        if self.external_electric_field_block:
            fdf["%block ExternalElectricField"] = self.external_electric_field_block

        # Slab dipole correction
        if self.slab_dipole_correction:
            fdf["Slab.DipoleCorrection"] = self.slab_dipole_correction
        if self.slab_dipole_correction_origin_block:
            fdf["%block Slab.DipoleCorrection.Origin"] = (
                self.slab_dipole_correction_origin_block
            )
        if self.slab_dipole_correction_vacuum_block:
            fdf["%block Slab.DipoleCorrection.Vacuum"] = (
                self.slab_dipole_correction_vacuum_block
            )

        # Geometry blocks
        if self.geometry_hartree_block:
            fdf["%block Geometry.Hartree"] = self.geometry_hartree_block
        if self.geometry_charge_block:
            fdf["%block Geometry.Charge"] = self.geometry_charge_block

        # Bulk bias
        if self.bulk_bias_voltage != 0.0:
            fdf["BulkBias.Voltage"] = f"{self.bulk_bias_voltage} eV"
            fdf["BulkBias.Tolerance"] = str(self.bulk_bias_tolerance)
            if self.bulk_bias_current:
                fdf["BulkBias.Current"] = "true"
            if self.bulk_bias_direction_block:
                fdf["%block BulkBias.Direction"] = self.bulk_bias_direction_block

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have charge/dipole/field parameters
        # These are SIESTA-specific for charged systems and transport
        return {}

    def generate_charge_dipole_block(self) -> None:
        """Generate the charge, dipole, and electric field options block."""
        logger.info("ChargeDipoleElectricField.generate_charge_dipole_block()")

        # Add comment header
        if self.comments:
            self.charge_dipole_fdf_arguments["#ChargeDipoleElectricField"] = (
                self.comments
            )

        # Net charge and doping
        if self.net_charge != 0:
            self.charge_dipole_fdf_arguments["NetCharge"] = f"{self.net_charge}"
        if self.simulate_doping:
            self.charge_dipole_fdf_arguments["SimulateDoping"] = (
                f"{self.simulate_doping}"
            )

        # Slab dipole correction
        if self.slab_dipole_correction:
            self.charge_dipole_fdf_arguments["Slab.DipoleCorrection"] = (
                f"{self.slab_dipole_correction}"
            )

        # Bulk bias
        if self.bulk_bias_voltage != 0.0:
            self.charge_dipole_fdf_arguments.update(
                {
                    "BulkBias.Voltage": f"{self.bulk_bias_voltage} eV",
                    "BulkBias.Tolerance": f"{self.bulk_bias_tolerance}",
                    "BulkBias.Current": f"{self.bulk_bias_current}",
                }
            )

    @classmethod
    def setup_charge_dipole_settings(
        cls,
        user_params: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG003
    ) -> "ChargeDipoleElectricField":
        """
        Create and configure a ChargeDipoleElectricField instance.

        This method handles proper key normalization, type conversion, and
        fuzzy matching to configure charge, dipole, and electric field settings
        from user parameters, with full parameter parsing.

        Args:
            user_params: Dictionary of user-defined parameters
                (case-insensitive, may include dots). If None or empty, all
                default values are used.
            **kwargs: Additional keyword arguments to override or supplement
                user_params.

        Returns
        -------
            ChargeDipoleElectricField: Configured instance with all fields set.
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]ChargeDipoleElectricField.setup_charge_dipole_settings()[/green]"
            )

        # Initialize instance with defaults
        instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default "
                    "Charge/Dipole values.[/blue]"
                )
            return instance

        # Get valid attribute names (lowercase for comparison)
        charge_attributes = {
            field.name.lower()
            for field in fields(cls)
            if not field.name.startswith("_") and field.name != "CONSOLE_VERBOSITY"
        }
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                f"[blue]Available ChargeDipoleElectricField attributes: "
                f"{charge_attributes}[/blue]"
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
            if key_normalized in charge_attributes:
                matched_attr = key_normalized
            else:
                # Fuzzy match: remove all underscores and compare
                key_no_underscores = key_normalized.replace("_", "")
                for attr in charge_attributes:
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
                # Dict fields
                if original_key in [
                    "external_electric_field_block",
                    "slab_dipole_correction_origin_block",
                    "slab_dipole_correction_vacuum_block",
                    "geometry_hartree_block",
                    "geometry_charge_block",
                    "bulk_bias_direction_block",
                ]:
                    if isinstance(value, dict):
                        setattr(instance, original_key, value)
                    elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            f"[yellow]Invalid type for {original_key}: "
                            f"expected dict, got {type(value)}[/yellow]"
                        )

                # Boolean fields
                elif original_key in ["simulate_doping", "bulk_bias_current"]:
                    bool_value = (
                        value.lower() in ("true", "t", "1", "yes")
                        if isinstance(value, str)
                        else value
                    )
                    setattr(instance, original_key, bool(bool_value))

                # Integer fields
                elif original_key == "net_charge":
                    try:
                        setattr(instance, original_key, int(value))
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[yellow]Could not convert "
                                f"{original_key}={value} to int[/yellow]"
                            )

                # Float fields
                elif original_key in ["bulk_bias_voltage", "bulk_bias_tolerance"]:
                    try:
                        setattr(instance, original_key, float(value))
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[yellow]Could not convert "
                                f"{original_key}={value} to float[/yellow]"
                            )

                # String fields
                elif original_key == "slab_dipole_correction":
                    setattr(instance, original_key, str(value))

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
                "[green]ChargeDipoleElectricField instance configured "
                "successfully.[/green]"
            )

        return instance
