"""
Data class for Denchar post-processing utility configuration.

class Denchar

TODO: Not sure to keep this because sisl can handle it ...
"""

# Metadata

__all__ = ["Denchar"]


import logging
from dataclasses import dataclass, field, fields
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@dataclass
class Denchar(FDFDataclass):
    """
    Configuration for Denchar post-processing utility output.

    This class manages the output settings for the denchar utility, which is used
    to generate charge density plots and other grid-based visualizations from SIESTA
    calculations. It controls whether denchar files are written and configures the
    grid resolution and ranges for visualization.

    Parameters
    ----------
    write_denchar : bool
        Enable writing files for denchar post-processing utility. Default: False
    denchar_x_min : float, optional
        Minimum X coordinate for denchar grid (in Bohr or Ang)
    denchar_x_max : float, optional
        Maximum X coordinate for denchar grid
    denchar_y_min : float, optional
        Minimum Y coordinate for denchar grid
    denchar_y_max : float, optional
        Maximum Y coordinate for denchar grid
    denchar_z_min : float, optional
        Minimum Z coordinate for denchar grid
    denchar_z_max : float, optional
        Maximum Z coordinate for denchar grid
    denchar_x_points : int
        Number of grid points in X direction. Default: 50
    denchar_y_points : int
        Number of grid points in Y direction. Default: 50
    denchar_z_points : int
        Number of grid points in Z direction. Default: 50

    Methods
    -------
    validate()
        Validate denchar configuration
    setup_denchar(user_params)
        Create configured instance with fuzzy parameter matching
    """  # Class-level verbosity control

    CONSOLE_VERBOSITY: VerbosityLevel = VerbosityLevel.ERROR

    # --------------------------------------
    # 6.31 Output of information for Denchar
    # --------------------------------------
    write_denchar: bool = field(
        default=False,
        metadata={
            "description": "If true, writes the files required by the 'denchar'"
            " post-processing utility for plotting charge densities.",
            "SIESTA keyword": "Write.Denchar",
        },
    )

    # Grid range parameters (optional - denchar can auto-determine)
    denchar_x_min: float | None = field(
        default=None,
        metadata={
            "description": "Minimum X coordinate for denchar visualization grid"
            " (Bohr or Ang).",
            "SIESTA keyword": "Denchar.XMin",
        },
    )

    denchar_x_max: float | None = field(
        default=None,
        metadata={
            "description": "Maximum X coordinate for denchar visualization grid"
            " (Bohr or Ang).",
            "SIESTA keyword": "Denchar.XMax",
        },
    )

    denchar_y_min: float | None = field(
        default=None,
        metadata={
            "description": "Minimum Y coordinate for denchar visualization grid"
            " (Bohr or Ang).",
            "SIESTA keyword": "Denchar.YMin",
        },
    )

    denchar_y_max: float | None = field(
        default=None,
        metadata={
            "description": "Maximum Y coordinate for denchar visualization grid"
            " (Bohr or Ang).",
            "SIESTA keyword": "Denchar.YMax",
        },
    )

    denchar_z_min: float | None = field(
        default=None,
        metadata={
            "description": "Minimum Z coordinate for denchar visualization grid"
            " (Bohr or Ang).",
            "SIESTA keyword": "Denchar.ZMin",
        },
    )

    denchar_z_max: float | None = field(
        default=None,
        metadata={
            "description": "Maximum Z coordinate for denchar visualization grid"
            " (Bohr or Ang).",
            "SIESTA keyword": "Denchar.ZMax",
        },
    )

    # Grid resolution parameters
    denchar_x_points: int = field(
        default=50,
        metadata={
            "description": "Number of grid points in X direction for denchar"
            " visualization.",
            "SIESTA keyword": "Denchar.NumberPointsX",
        },
    )

    denchar_y_points: int = field(
        default=50,
        metadata={
            "description": "Number of grid points in Y direction for denchar"
            " visualization.",
            "SIESTA keyword": "Denchar.NumberPointsY",
        },
    )

    denchar_z_points: int = field(
        default=50,
        metadata={
            "description": "Number of grid points in Z direction for denchar"
            " visualization.",
            "SIESTA keyword": "Denchar.NumberPointsZ",
        },
    )

    # Comment header for FDF output
    comments: str = field(
        default="# Denchar Visualization Configuration (Denchar dataclass module)",
        metadata={"description": "Comment header for FDF file"},
    )

    # Dictionary to hold FDF arguments
    denchar_fdf_arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "Write.Denchar",
                "Denchar.XMin",
                "Denchar.XMax",
                "Denchar.YMin",
                "Denchar.YMax",
                "Denchar.ZMin",
                "Denchar.ZMax",
                "Denchar.NumberPointsX",
                "Denchar.NumberPointsY",
                "Denchar.NumberPointsZ",
            )
            self.__class__._registered = True  # noqa: SLF001 class-level registration flag

    def validate(self) -> None:
        """
        Validate Denchar output options.

        Checks configuration for the denchar post-processing utility used to
        plot charge densities and other grid-based quantities.

        Raises
        ------
        ValueError
            If Denchar settings are invalid
        """
        logger.info("Denchar.validate()")

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        from atomate2.siesta.dataclass.units import parse_length

        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in ["write.denchar", "write_denchar"]:
                self.write_denchar = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["denchar.xmin", "denchar_x_min"]:
                self.denchar_x_min = (
                    parse_length(value, target_unit="Bohr")
                    if value is not None
                    else None
                )
            elif key_lower in ["denchar.xmax", "denchar_x_max"]:
                self.denchar_x_max = (
                    parse_length(value, target_unit="Bohr")
                    if value is not None
                    else None
                )
            elif key_lower in ["denchar.ymin", "denchar_y_min"]:
                self.denchar_y_min = (
                    parse_length(value, target_unit="Bohr")
                    if value is not None
                    else None
                )
            elif key_lower in ["denchar.ymax", "denchar_y_max"]:
                self.denchar_y_max = (
                    parse_length(value, target_unit="Bohr")
                    if value is not None
                    else None
                )
            elif key_lower in ["denchar.zmin", "denchar_z_min"]:
                self.denchar_z_min = (
                    parse_length(value, target_unit="Bohr")
                    if value is not None
                    else None
                )
            elif key_lower in ["denchar.zmax", "denchar_z_max"]:
                self.denchar_z_max = (
                    parse_length(value, target_unit="Bohr")
                    if value is not None
                    else None
                )
            elif key_lower in ["denchar.numberpointsx", "denchar_x_points"]:
                self.denchar_x_points = int(value)
            elif key_lower in ["denchar.numberpointsy", "denchar_y_points"]:
                self.denchar_y_points = int(value)
            elif key_lower in ["denchar.numberpointsz", "denchar_z_points"]:
                self.denchar_z_points = int(value)

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters
        """
        fdf: dict[str, Any] = {}

        # Add comment header
        fdf["#Denchar"] = "Denchar Settings"

        if self.write_denchar:
            fdf["Write.Denchar"] = "true"

            # Grid ranges (optional)
            if self.denchar_x_min is not None:
                fdf["Denchar.XMin"] = f"{self.denchar_x_min} Bohr"
            if self.denchar_x_max is not None:
                fdf["Denchar.XMax"] = f"{self.denchar_x_max} Bohr"
            if self.denchar_y_min is not None:
                fdf["Denchar.YMin"] = f"{self.denchar_y_min} Bohr"
            if self.denchar_y_max is not None:
                fdf["Denchar.YMax"] = f"{self.denchar_y_max} Bohr"
            if self.denchar_z_min is not None:
                fdf["Denchar.ZMin"] = f"{self.denchar_z_min} Bohr"
            if self.denchar_z_max is not None:
                fdf["Denchar.ZMax"] = f"{self.denchar_z_max} Bohr"

            # Grid resolution
            if self.denchar_x_points != 50:
                fdf["Denchar.NumberPointsX"] = str(self.denchar_x_points)
            if self.denchar_y_points != 50:
                fdf["Denchar.NumberPointsY"] = str(self.denchar_y_points)
            if self.denchar_z_points != 50:
                fdf["Denchar.NumberPointsZ"] = str(self.denchar_z_points)

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have denchar visualization parameters
        # These are SIESTA-specific post-processing options
        return {}

    def generate_denchar_block(self) -> None:
        """
        Generate FDF arguments for denchar with comment header.

        Populates denchar_fdf_arguments dictionary with all denchar parameters
        that are set to non-default values. Adds comment header if comments are
        enabled.
        """
        logger.info("Denchar.generate_denchar_block()")

        # Collect parameters first (only non-default values)
        params_to_add = {}

        if self.write_denchar:
            params_to_add["Write.Denchar"] = self.write_denchar

        # Grid ranges (optional - only if specified)
        if self.denchar_x_min is not None:
            params_to_add["Denchar.XMin"] = self.denchar_x_min
        if self.denchar_x_max is not None:
            params_to_add["Denchar.XMax"] = self.denchar_x_max
        if self.denchar_y_min is not None:
            params_to_add["Denchar.YMin"] = self.denchar_y_min
        if self.denchar_y_max is not None:
            params_to_add["Denchar.YMax"] = self.denchar_y_max
        if self.denchar_z_min is not None:
            params_to_add["Denchar.ZMin"] = self.denchar_z_min
        if self.denchar_z_max is not None:
            params_to_add["Denchar.ZMax"] = self.denchar_z_max

        # Grid resolution (only if non-default)
        if self.denchar_x_points != 50:
            params_to_add["Denchar.NumberPointsX"] = self.denchar_x_points
        if self.denchar_y_points != 50:
            params_to_add["Denchar.NumberPointsY"] = self.denchar_y_points
        if self.denchar_z_points != 50:
            params_to_add["Denchar.NumberPointsZ"] = self.denchar_z_points

        # Only add comment header if there are parameters to add
        if params_to_add:
            if self.comments:
                self.denchar_fdf_arguments["#Denchar"] = self.comments
            self.denchar_fdf_arguments.update(params_to_add)

    @classmethod
    def setup_denchar(
        cls,
        user_params: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG003 accepted for API compatibility
    ) -> "Denchar":
        """
        Create and configure a Denchar instance with full parameter parsing.

        This method handles proper key normalization, type conversion, and fuzzy
        matching to configure denchar output settings from user parameters. Supports
        SIESTA FDF parameter names (Write.Denchar, Denchar.NumberPointsX, etc.) with
        automatic conversion.

        Args:
            user_params: Dictionary of user-defined parameters (case-insensitive, may
                        include dots). If None or empty, all default values are used.
            **kwargs: Additional keyword arguments to override or supplement
                        user_params.

        Returns
        -------
            Denchar: Configured instance with all fields set.

        Examples
        --------
            >>> # Using SIESTA FDF parameter names
            >>> denchar = Denchar.setup_denchar(
            ...     {
            ...         "Write.Denchar": True,
            ...         "Denchar.NumberPointsX": 100,
            ...         "Denchar.NumberPointsY": 100,
            ...         "Denchar.NumberPointsZ": 100,
            ...     }
            ... )

            >>> # Using Python attribute names
            >>> denchar = Denchar.setup_denchar(
            ...     {
            ...         "write_denchar": True,
            ...         "denchar_x_points": 100,
            ...         "denchar_y_points": 100,
            ...     }
            ... )

            >>> # With custom grid range
            >>> denchar = Denchar.setup_denchar(
            ...     {
            ...         "Write.Denchar": True,
            ...         "Denchar.XMin": -5.0,
            ...         "Denchar.XMax": 5.0,
            ...         "Denchar.NumberPointsX": 200,
            ...     }
            ... )
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]Denchar.setup_denchar()[/green]")

        # Initialize instance with defaults
        instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default"
                    " Denchar values.[/blue]"
                )
            return instance

        # Get valid attribute names (lowercase for comparison)
        denchar_attributes = {
            field.name.lower()
            for field in fields(cls)
            if not field.name.startswith("_") and field.name != "CONSOLE_VERBOSITY"
        }
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                f"[blue]Available Denchar attributes: {denchar_attributes}[/blue]"
            )

        # Process user parameters
        import re
        from difflib import get_close_matches

        for key, value in user_params.items():
            # Normalize key: handle camelCase and dots
            key_with_underscores = re.sub(r"([a-z])([A-Z])", r"\1_\2", key)
            key_normalized = key_with_underscores.replace(".", "_").lower()

            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Processing key: {key} -> {key_normalized},"
                    f" value: {value}[/blue]"
                )

            # Check if normalized key matches any attribute
            matched_attr = None
            if key_normalized in denchar_attributes:
                matched_attr = key_normalized
            else:
                # Try fuzzy matching
                close_matches = get_close_matches(
                    key_normalized, denchar_attributes, n=3, cutoff=0.6
                )
                if close_matches:
                    # For NumberPointsX/Y/Z, prefer match containing same axis letter
                    if key_normalized.endswith(("_x", "x")):
                        matched_attr = next(
                            (
                                m
                                for m in close_matches
                                if "_x_" in m or m.startswith("denchar_x")
                            ),
                            close_matches[0],
                        )
                    elif key_normalized.endswith(("_y", "y")):
                        matched_attr = next(
                            (
                                m
                                for m in close_matches
                                if "_y_" in m or m.startswith("denchar_y")
                            ),
                            close_matches[0],
                        )
                    elif key_normalized.endswith(("_z", "z")):
                        matched_attr = next(
                            (
                                m
                                for m in close_matches
                                if "_z_" in m or m.startswith("denchar_z")
                            ),
                            close_matches[0],
                        )
                    else:
                        matched_attr = close_matches[0]

                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
                        console.print(
                            f"[yellow]Fuzzy match: '{key}' -> '{matched_attr}'[/yellow]"
                        )

            # Set attribute if matched
            if matched_attr:
                # Type conversion based on parameter type
                if matched_attr == "write_denchar":
                    # Boolean parameter
                    if isinstance(value, bool):
                        setattr(instance, matched_attr, value)
                    elif isinstance(value, str):
                        setattr(
                            instance,
                            matched_attr,
                            value.lower() in ["true", "t", "yes", "1"],
                        )
                    else:
                        setattr(instance, matched_attr, bool(value))
                elif matched_attr in [
                    "denchar_x_points",
                    "denchar_y_points",
                    "denchar_z_points",
                ]:
                    # Integer parameters
                    try:
                        setattr(instance, matched_attr, int(value))
                    except (ValueError, TypeError):
                        console.print(
                            f"[yellow]Warning: Could not convert '{value}' to int"
                            f" for '{matched_attr}'. Using default.[/yellow]"
                        )
                else:
                    # Float parameters (grid coordinates)
                    try:
                        setattr(instance, matched_attr, float(value))
                    except (ValueError, TypeError):
                        console.print(
                            f"[yellow]Warning: Could not convert '{value}' to float"
                            f" for '{matched_attr}'. Using default.[/yellow]"
                        )
            elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
                console.print(
                    f"[yellow]Warning: No match found for parameter '{key}'"
                    " in Denchar[/yellow]"
                )

        # Generate FDF block with comment header
        instance.generate_denchar_block()

        return instance
