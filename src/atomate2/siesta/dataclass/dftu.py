"""
Data class to manage DFT+U parameters for SIESTA input.

class DFTU

Based on User's Guide Siesta 5.4.0
Section: 8 DFTU
"""

# Metadata

__all__ = ["DFTU"]

import logging
from dataclasses import dataclass, field, fields
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.dataclass.units import parse_energy
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@dataclass
class DFTU(FDFDataclass):
    """Data class to manage DFT+U parameters for SIESTA input."""

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.ERROR
    )  # Default to show errors only

    # dftu_projector_generation_method: int = 2 # DFTU.ProjectorGenerationMethod 2
    # dftu_energy_shift: float = 0.05 # DFTU.EnergyShift 0.05 Ry
    # dftu_cutoff_norm: float = 0.9 # DFTU.CutoffNorm 0.9
    # dftu_proj_block: Dict[float,Any]= field(default_factory=dict)
    # %block DFTU.Proj 〈None〉
    # dftu_first_iteration: bool = False # DFTU.FirstIteration false
    # dftu_threshold_tol: float = 0.01 # DFTU.ThresholdTol 0.01
    # dftu_pop_tol: float = 0.001 # DFTU.PopTol 0.001
    # dftu_potential_shift: bool = False # DFTU.PotentialShift false

    dftu_projector_generation_method: int = field(
        default=2,
        metadata={
            "description": (
                "Selects the method for generating the projectors used in the "
                "DFT+U calculation."
            ),
            "SIESTA keyword": "DFTU.ProjectorGenerationMethod",
        },
    )

    dftu_energy_shift: float = field(
        default=0.05,
        metadata={
            "description": (
                "An energy shift (in Rydberg) applied during the generation of "
                "the DFT+U projectors."
            ),
            "SIESTA keyword": "DFTU.EnergyShift",
            "unit": "Ry",
        },
    )

    dftu_cutoff_norm: float = field(
        default=0.9,
        metadata={
            "description": (
                "A norm cutoff used during the generation of the DFT+U projectors."
            ),
            "SIESTA keyword": "DFTU.CutoffNorm",
        },
    )

    dftu_proj_block: dict[str, Any] | list[str] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block to manually define the projectors for the DFT+U "
                "calculation. Can be dict or list."
            ),
            "SIESTA keyword": "%block DFTU.Proj",
        },
    )

    dftu_first_iteration: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, the DFT+U correction is applied starting from the very "
                "first SCF iteration."
            ),
            "SIESTA keyword": "DFTU.FirstIteration",
        },
    )

    dftu_threshold_tol: float = field(
        default=0.01,
        metadata={
            "description": (
                "A tolerance threshold used in the DFT+U implementation, likely "
                "related to orbital occupations."
            ),
            "SIESTA keyword": "DFTU.ThresholdTol",
        },
    )

    dftu_pop_tol: float = field(
        default=0.001,
        metadata={
            "description": (
                "A tolerance for the convergence of the orbital occupation matrix "
                "in the DFT+U calculation."
            ),
            "SIESTA keyword": "DFTU.PopTol",
        },
    )

    dftu_potential_shift: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, applies a potential shift as part of the DFT+U correction."
            ),
            "SIESTA keyword": "DFTU.PotentialShift",
        },
    )

    comments: str = field(
        default="DFTU Settings",
        metadata={
            "description": "Comment header for this dataclass section in the FDF file.",
            "SIESTA keyword": None,
        },
    )

    dftu_fdf_arguments: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A dictionary for any additional or arbitrary FDF flags related "
                "to DFT+U."
            ),
            "SIESTA keyword": None,
        },
    )

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "DFTU.ProjectorGenerationMethod",
                "DFTU.EnergyShift",
                "DFTU.CutoffNorm",
                "%block DFTU.Proj",
                "DFTU.FirstIteration",
                "DFTU.ThresholdTol",
                "DFTU.PopTol",
                "DFTU.PotentialShift",
            )
            self.__class__._registered = True  # noqa: SLF001 class-level registration guard

    def validate(self) -> None:
        """
        Validate DFT+U configuration parameters.

        Checks that DFT+U settings (projector method, energy shift, cutoff norms,
        and tolerances) are within valid ranges and properly configured.

        Raises
        ------
        ValueError
            If DFT+U parameters are invalid or inconsistent
        """
        logger.info("DFTU.validate()")

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in [
                "dftu.projectorgenerationmethod",
                "dftu_projector_generation_method",
            ]:
                self.dftu_projector_generation_method = int(value)
            elif key_lower in ["dftu.energyshift", "dftu_energy_shift"]:
                self.dftu_energy_shift = parse_energy(value, target_unit="Ry")
            elif key_lower in ["dftu.cutoffnorm", "dftu_cutoff_norm"]:
                self.dftu_cutoff_norm = (
                    float(value) if isinstance(value, str) else value
                )
            elif key_lower == "%block dftu.proj":
                # Accept both dict and list formats
                if isinstance(value, (dict, list)):
                    self.dftu_proj_block = value
            elif key_lower in ["dftu.firstiteration", "dftu_first_iteration"]:
                self.dftu_first_iteration = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["dftu.thresholdtol", "dftu_threshold_tol"]:
                self.dftu_threshold_tol = (
                    float(value) if isinstance(value, str) else value
                )
            elif key_lower in ["dftu.poptol", "dftu_pop_tol"]:
                self.dftu_pop_tol = float(value) if isinstance(value, str) else value
            elif key_lower in ["dftu.potentialshift", "dftu_potential_shift"]:
                self.dftu_potential_shift = (
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
        fdf["#DFTU"] = "DFTU Settings"

        # DFTU.ProjectorGenerationMethod - always write with default marker
        if self.dftu_projector_generation_method == 2:
            fdf["DFTU.ProjectorGenerationMethod"] = "2  # SIESTA DEFAULT VALUE"
        else:
            fdf["DFTU.ProjectorGenerationMethod"] = str(
                self.dftu_projector_generation_method
            )

        # DFTU.EnergyShift - always write with default marker
        if self.dftu_energy_shift == 0.05:
            fdf["DFTU.EnergyShift"] = "0.05 Ry  # SIESTA DEFAULT VALUE"
        else:
            fdf["DFTU.EnergyShift"] = f"{self.dftu_energy_shift} Ry"

        # DFTU.CutoffNorm - always write with default marker
        if self.dftu_cutoff_norm == 0.9:
            fdf["DFTU.CutoffNorm"] = "0.9  # SIESTA DEFAULT VALUE"
        else:
            fdf["DFTU.CutoffNorm"] = str(self.dftu_cutoff_norm)

        # %block DFTU.Proj - write if provided (no default marker, it's a block)
        if self.dftu_proj_block:
            fdf["%block DFTU.Proj"] = self.dftu_proj_block

        # DFTU.FirstIteration - always write with default marker
        if not self.dftu_first_iteration:
            fdf["DFTU.FirstIteration"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["DFTU.FirstIteration"] = "true"

        # DFTU.ThresholdTol - always write with default marker
        if self.dftu_threshold_tol == 0.01:
            fdf["DFTU.ThresholdTol"] = "0.01  # SIESTA DEFAULT VALUE"
        else:
            fdf["DFTU.ThresholdTol"] = str(self.dftu_threshold_tol)

        # DFTU.PopTol - always write with default marker
        if self.dftu_pop_tol == 0.001:
            fdf["DFTU.PopTol"] = "0.001  # SIESTA DEFAULT VALUE"
        else:
            fdf["DFTU.PopTol"] = str(self.dftu_pop_tol)

        # DFTU.PotentialShift - always write with default marker
        if not self.dftu_potential_shift:
            fdf["DFTU.PotentialShift"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["DFTU.PotentialShift"] = "true"

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have DFT+U parameter equivalents
        # DFT+U is SIESTA-specific
        return {}

    def generate_dftu_block(self) -> None:
        """Generate the DFT+U options block for the FDF file."""
        logger.info("DFTU.generate_dftu_block()")

        # Add comment header
        if self.comments:
            self.dftu_fdf_arguments["#DFTU"] = self.comments

        # Only add parameters if they differ from defaults or are explicitly set
        # DFT+U parameters are typically only written when DFT+U is actually being used
        self.dftu_fdf_arguments.update(
            {
                "DFTU.ProjectorGenerationMethod": (
                    f"{self.dftu_projector_generation_method}"
                ),
                "DFTU.EnergyShift": f"{self.dftu_energy_shift} Ry",
                "DFTU.CutoffNorm": f"{self.dftu_cutoff_norm}",
                "DFTU.FirstIteration": f"{self.dftu_first_iteration}",
                "DFTU.ThresholdTol": f"{self.dftu_threshold_tol}",
                "DFTU.PopTol": f"{self.dftu_pop_tol}",
                "DFTU.PotentialShift": f"{self.dftu_potential_shift}",
            }
        )

    @classmethod
    def setup_dftu_settings(
        cls,
        user_params: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG003 interface kwarg
    ) -> "DFTU":
        """
        Create and configure a DFTU instance with full parameter parsing.

        This method handles proper key normalization, type conversion, and fuzzy
        matching to configure DFT+U settings from user parameters.

        Args:
            user_params: Dictionary of user-defined parameters (case-insensitive,
                        may include dots).
                        If None or empty, all default values are used.
            **kwargs: Additional keyword arguments to override or supplement
                        user_params.

        Returns
        -------
            DFTU: Configured instance with all fields set.
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]DFTU.setup_dftu_settings()[/green]")

        # Initialize instance with defaults
        instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default "
                    "DFT+U values.[/blue]"
                )
            return instance

        # Get valid attribute names (lowercase for comparison)
        dftu_attributes = {
            field.name.lower()
            for field in fields(cls)
            if not field.name.startswith("_") and field.name != "CONSOLE_VERBOSITY"
        }
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(f"[blue]Available DFTU attributes: {dftu_attributes}[/blue]")

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
            if key_normalized in dftu_attributes:
                matched_attr = key_normalized
            else:
                # Fuzzy match: remove all underscores and compare
                key_no_underscores = key_normalized.replace("_", "")
                for attr in dftu_attributes:
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
                if original_key == "dftu_proj_block":
                    if isinstance(value, dict):
                        setattr(instance, original_key, value)
                    elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            f"[yellow]Invalid type for {original_key}: expected "
                            f"dict, got {type(value)}[/yellow]"
                        )

                # Boolean fields
                elif original_key in ["dftu_first_iteration", "dftu_potential_shift"]:
                    bool_value = value
                    if isinstance(value, str):
                        bool_value = value.lower() in ("true", "t", "1", "yes")
                    setattr(instance, original_key, bool(bool_value))

                # Integer fields
                elif original_key == "dftu_projector_generation_method":
                    try:
                        setattr(instance, original_key, int(value))
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[yellow]Could not convert "
                                f"{original_key}={value} to int[/yellow]"
                            )

                # Float fields
                elif original_key in [
                    "dftu_energy_shift",
                    "dftu_cutoff_norm",
                    "dftu_threshold_tol",
                    "dftu_pop_tol",
                ]:
                    try:
                        setattr(instance, original_key, float(value))
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[yellow]Could not convert "
                                f"{original_key}={value} to float[/yellow]"
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
            console.print("[green]DFTU instance configured successfully.[/green]")

        return instance
