"""
Module defining base SIESTA input set and generator.

class ExchangeCorrelationFunctionals

Based on User's Guide Siesta 5.4.0
Section: 6.6 Exchange-correlation functionals
"""

# Metadata

__all__ = ["ExchangeCorrelationFunctionals"]

import logging
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@dataclass
class ExchangeCorrelationFunctionals(FDFDataclass):
    """Data class to manage exchange-correlation functionals for SIESTA input."""

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.INFO
    )  # Default to show info messages

    _comments: str = field(
        default="ExchangeCorrelationFunctionals",
        metadata={
            "description": (
                "User-provided comments to be included as a comment block "
                "in the FDF file."
            ),
            "SIESTA keyword": None,
        },
    )
    _user_params: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Store original user parameters for validation",
            "SIESTA keyword": None,
        },
    )
    xc_functional: str = field(
        default="GGA",
        metadata={
            "description": (
                "The general family of the exchange-correlation functional "
                "(e.g., LDA, GGA) to be used."
            ),
            "SIESTA keyword": "XC.functional",
        },
    )
    xc_authors: str = field(
        default="PBE",
        metadata={
            "description": (
                "The specific parametrization or 'author' of the chosen "
                "exchange-correlation functional (e.g., PW91, PBE, revPBE)."
            ),
            "SIESTA keyword": "XC.authors",
        },
    )
    xc_use_bsc_cell_xc: bool = field(
        default=False,
        metadata={
            "description": (
                "A specific flag used within some van der Waals functionals "
                "(like vdW-DF-cx) to modify the cell-dependent part of the "
                "correlation."
            ),
            "SIESTA keyword": "XC.Use.BSC.CellXC",
        },
    )
    xc_block: dict[str, Any] | None = field(
        default_factory=dict,
        metadata={
            "description": (
                "A block for detailed specification of the XC functional, "
                "particularly for hybrid or complex vdW functionals."
            ),
            "SIESTA keyword": "%block XC.Mix",
        },
    )
    xc_fdf_arguments: OrderedDict[str, Any] = field(
        default_factory=OrderedDict,
        metadata={
            "description": (
                "A dictionary for any additional or arbitrary FDF flags "
                "related to the XC functional."
            ),
            "SIESTA keyword": None,
        },
    )

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "XC.functional",
                "XC.authors",
                "XC.Use.BSC.CellXC",
                "%block XC.Mix",
            )
            self.__class__._registered = True  # noqa: SLF001 class-level flag

    @classmethod
    def setup_xc_settings(
        cls, user_params: dict[str, Any] | None = None
    ) -> "ExchangeCorrelationFunctionals":
        """
        Create and configure an ExchangeCorrelationFunctionals instance.

        Based on user parameters, retaining all default values for unspecified
        fields. Issues warnings for invalid keys and skips them.

        Args:
            user_params (dict, optional): Dictionary of user-defined parameters
                (case-insensitive, may include dots). If None or empty, all
                default ExchangeCorrelationFunctionals values are used.

        Returns
        -------
            ExchangeCorrelationFunctionals: Configured instance with all fields
                (default and user-specified) and FDF arguments.
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]ExchangeCorrelationFunctionals.setup_xc_settings()[/green]"
            )

        # Initialize instance with defaults
        xc_settings_instance = cls()

        # Store user_params for validate method
        xc_settings_instance._user_params = user_params or {}

        # Handle case where user_params is None or empty
        if not xc_settings_instance._user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default "
                    "ExchangeCorrelationFunctionals values.[/blue]"
                )
                console.print(
                    f"[blue]user_params: {xc_settings_instance._user_params}[/blue]"
                )
        else:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]user_params: {xc_settings_instance._user_params}[/blue]"
                )

            # Get valid attribute names (lowercase for comparison),
            # excluding _comments and _user_params
            xc_settings_attributes = {
                field.name.lower()
                for field in fields(cls)
                if not field.name.startswith("_")
            }
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Available ExchangeCorrelationFunctionals attributes: "
                    f"{xc_settings_attributes}[/blue]"
                )

            # Process user parameters
            for key, value in xc_settings_instance._user_params.items():
                # Normalize key: convert to lowercase and replace dots with underscores
                key_normalized = key.lower().replace(".", "_")
                if cls.CONSOLE_VERBOSITY.value == VerbosityLevel.DEBUG.value:
                    console.print(
                        f"[blue]Processing key: {key} -> normalized: "
                        f"{key_normalized}, value: {value}[/blue]"
                    )

                # Skip _comments and _user_params if provided by user
                if key_normalized in ["_comments", "_user_params"]:
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            f"[yellow]Ignoring user-provided '{key}'; "
                            f"it is internal.[/yellow]"
                        )
                    continue

                # Check if normalized key matches any attribute
                if key_normalized in xc_settings_attributes:
                    # Find the original attribute name (preserving case)
                    original_key = next(
                        field.name
                        for field in fields(cls)
                        if field.name.lower() == key_normalized
                    )
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                        console.print(
                            f"[blue]Matched ExchangeCorrelationFunctionals field: "
                            f"{original_key} = {value}[/blue]"
                        )

                    # Handle type conversion for specific fields
                    if original_key == "xc_block" and isinstance(value, dict):
                        setattr(xc_settings_instance, original_key, value)
                    elif original_key == "xc_use_bsc_cell_xc":
                        bool_value = (
                            value.lower() in ("true", "t", "1", "yes")
                            if isinstance(value, str)
                            else value
                        )
                        setattr(xc_settings_instance, original_key, bool(bool_value))
                    elif original_key in ["xc_functional", "xc_authors"]:
                        setattr(
                            xc_settings_instance,
                            original_key,
                            value.upper() if isinstance(value, str) else value,
                        )
                    else:
                        setattr(xc_settings_instance, original_key, value)
                elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                    console.print(
                        f"[yellow]Key '{key}' does not match any "
                        f"ExchangeCorrelationFunctionals field, skipping.[/yellow]"
                    )

        # Validate settings
        try:
            xc_settings_instance.validate()
        except ValueError as e:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.ERROR.value:
                console.print(f"[red]Validation failed: {e}[/red]")
            raise

        # Generate FDF block
        xc_settings_instance.generate()

        # Clear _user_params after validation to avoid memory leaks
        xc_settings_instance._user_params = None

        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(
                "[green]Validation & Generation: "
                "[yellow]ExchangeCorrelationFunctionals[/yellow] Successful![/green]"
            )

        return xc_settings_instance

    def validate(self) -> None:
        """
        Validate the exchange-correlation functional settings.

        Includes checks for xc, xc.functional, and xc.authors.

        Raises
        ------
            ValueError: If xc_functional or xc_authors are invalid.
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]ExchangeCorrelationFunctionals.validate()[/green]")

        # Check for xc, xc.functional, and xc.authors in _user_params
        if self._user_params:
            # Check for 'xc' key
            if (
                any(k.lower() == "xc" for k in self._user_params)
                and self.CONSOLE_VERBOSITY.value >= VerbosityLevel.ERROR.value
            ):
                raise ValueError("Use xc.functional and xc.authors instead of 'xc'")

            # Check for xc.functional and xc.authors
            has_xc_functional = any(
                k.lower() in ["xc.functional", "xc_functional"]
                for k in self._user_params
            )
            has_xc_authors = any(
                k.lower() in ["xc.authors", "xc_authors"] for k in self._user_params
            )
            if has_xc_functional and has_xc_authors:
                if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                    console.print(
                        "[blue]Successfully validated XC.Functional and "
                        "XC.Authors[/blue]"
                    )
            elif has_xc_functional or has_xc_authors:
                if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                    console.print(
                        "[yellow]Warning: Both xc.functional and xc.authors "
                        "must be specified together[/yellow]"
                    )
            elif self.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                console.print(
                    "[yellow]Warning: Default values are taken for "
                    "XC.Functional and XC.Authors[/yellow]"
                )

        # Validate xc_functional and xc_authors
        allowed_xc_functionals = ["LDA", "LSD", "GGA", "VDW"]
        if self.xc_functional.upper() not in allowed_xc_functionals:
            raise ValueError(
                f"Invalid functional type '{self.xc_functional}'. "
                f"Allowed values are: {allowed_xc_functionals}"
            )

        allowed_xc_authors = [
            "CA",
            "PW92",
            "PW91",
            "PBE",
            "REVPBE",
            "RPBE",
            "WC",
            "AM05",
            "PBESOL",  # Note: Will be written as "PBEsol" in FDF (SIESTA format)
            "PBEJSJELO",
            "PBEJSJRHEG",
            "PBEGCGXLO",
            "PBEGCGXHEG",
            "BLYP",
            "DRSLL",
            "LMKLL",
            "KBM",
            "C09",
            "BH",
            "VV",
        ]
        if self.xc_authors.upper() not in allowed_xc_authors:
            raise ValueError(
                f"Invalid xc authors type '{self.xc_authors}'. "
                f"Allowed values are: {allowed_xc_authors}"
            )

        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]Validation: "
                "[yellow]ExchangeCorrelationFunctionals[/yellow] Successful![/green]"
            )

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]ExchangeCorrelationFunctionals.update_from_fdf()[/green]"
            )

        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in ["xc.functional", "xc_functional"]:
                self.xc_functional = str(value).upper()
            elif key_lower in ["xc.authors", "xc_authors"]:
                self.xc_authors = str(value).upper()
            elif key_lower in ["xc.use.bsc.cellxc", "xc_use_bsc_cellxc"]:
                self.xc_use_bsc_cell_xc = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower == "%block xc.mix" and isinstance(value, dict):
                self.xc_block = value

    @staticmethod
    def _format_xc_authors_for_siesta(xc_authors: str) -> str:
        """
        Format XC.Authors for SIESTA FDF file.

        SIESTA expects specific capitalization (e.g., "PBEsol" not "PBESOL").

        Args:
            xc_authors: XC authors string

        Returns
        -------
            Properly formatted XC.Authors for SIESTA
        """
        xc_upper = xc_authors.upper()

        # Special cases that need mixed case
        if xc_upper == "PBESOL":
            return "PBEsol"
        if xc_upper == "REVPBE":
            return "revPBE"

        # All others use uppercase
        return xc_upper

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]ExchangeCorrelationFunctionals.generate_fdf()[/green]"
            )

        fdf: dict[str, Any] = OrderedDict()
        fdf["#ExchangeCorrelationFunctionals"] = "ExchangeCorrelationFunctionals"

        # XC.Functional - always write with default marker
        if self.xc_functional.upper() == "GGA":
            fdf["XC.Functional"] = (
                f"{self.xc_functional.upper()}  # SIESTA DEFAULT VALUE"
            )
        else:
            fdf["XC.Functional"] = self.xc_functional.upper()

        # XC.Authors - always write with default marker
        formatted_authors = self._format_xc_authors_for_siesta(self.xc_authors)
        if formatted_authors == "PBE":
            fdf["XC.Authors"] = f"{formatted_authors}  # SIESTA DEFAULT VALUE"
        else:
            fdf["XC.Authors"] = formatted_authors

        # XC.Use.BSC.CellXC - always write with default marker
        if not self.xc_use_bsc_cell_xc:
            fdf["XC.Use.BSC.CellXC"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["XC.Use.BSC.CellXC"] = "true"

        if self.xc_block:
            fdf["%block XC.Mix"] = self.xc_block

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE uses 'xc' parameter (combines functional + authors)
        return {"xc": f"{self.xc_functional.upper()}:{self.xc_authors.upper()}"}

    def generate(self) -> None:
        """Generate the exchange-correlation functional block for the FDF file."""
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]ExchangeCorrelationFunctionals.generate()[/green]")

        self.xc_fdf_arguments = OrderedDict()

        # Add comments if provided
        if self._comments:
            self.xc_fdf_arguments["#ExchangeCorrelationFunctionals"] = self._comments

        # Add XC functional and authors
        self.xc_fdf_arguments["XC.Functional"] = self.xc_functional.upper()
        self.xc_fdf_arguments["XC.Authors"] = self._format_xc_authors_for_siesta(
            self.xc_authors
        )

        # Add XC.Use.BSC.CellXC if True
        if self.xc_use_bsc_cell_xc:
            self.xc_fdf_arguments["XC.Use.BSC.CellXC"] = ".true."

        # Add xc_block if provided
        if self.xc_block:
            if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.ERROR.value:
                console.print("[red]XC block not implemented yet![/red]")
            raise ValueError("XC block not implemented yet!")
