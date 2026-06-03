"""
Module defining NetCDF (CDF4) output options for SIESTA calculations.

This module provides configuration for NetCDF format output including
compression, precision, and MPI parallel I/O settings for grid-based data.

class NetcdfOptions

Based on User's Guide Siesta 5.4.0
Section:  6.32 NetCDF (CDF4) output file
"""

# Metadata

__all__ = ["NetcdfOptions"]

from dataclasses import dataclass, field, fields
from typing import Dict, Any, Optional

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

import logging

logger = logging.getLogger(__name__)


@dataclass
class NetcdfOptions(FDFDataclass):
    """
    Configuration for NetCDF (CDF4) format output in SIESTA.

    This class manages NetCDF output settings for grid-based data including
    charge densities, potentials, and other quantities. NetCDF provides
    portable, self-describing binary format with optional compression.

    Parameters
    ----------
    cdf_save : bool
        Enable saving grid-based data in NetCDF format. Default: False
    cdf_compress : int
        Compression level for NetCDF files (0-9, 0=none). Default: 0
    cdf_mpi : bool
        Use parallel NetCDF I/O for MPI runs. Default: False
    cdf_grid_precision : str
        Precision for grid data ('single' or 'double'). Default: 'single'

    Methods
    -------
    validate()
        Validate NetCDF output configuration
    setup_netcdf_settings(user_params)
        Create configured instance with fuzzy parameter matching
    """

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = VerbosityLevel.ERROR

    # ------------------------------
    # 6.32 NetCDF (CDF4) output file
    # ------------------------------
    cdf_save: bool = field(
        default=False,
        metadata={
            "description": "A master flag to enable saving of grid-based data (like charge density) in the NetCDF format.",
            "SIESTA keyword": "CDF.Save",
        },
    )

    cdf_compress: int = field(
        default=0,
        metadata={
            "description": "Sets the compression level (0-9) for NetCDF files. A value of 0 means no compression.",
            "SIESTA keyword": "CDF.Compress",
        },
    )

    cdf_mpi: bool = field(
        default=False,
        metadata={
            "description": "If true, uses the parallel I/O capabilities of the NetCDF library for faster file writing in MPI runs.",
            "SIESTA keyword": "CDF.MPI",
        },
    )

    cdf_grid_precision: str = field(
        default="single",
        metadata={
            "description": "Sets the precision for grid data written to NetCDF files. Options are 'single' or 'double'.",
            "SIESTA keyword": "CDF.Grid.Precision",
        },
    )

    # Comment header for FDF output
    comments: str = field(
        default="# NetCDF Output Configuration (NetcdfOptions dataclass module)",
        metadata={"description": "Comment header for FDF file"},
    )

    # Dictionary to hold FDF arguments
    netcdf_fdf_arguments: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "CDF.Save",
                "CDF.Compress",
                "CDF.MPI",
                "CDF.Grid.Precision",
            )
            self.__class__._registered = True

    def validate(self):
        """
        Validate NetCDF output options.

        Checks that NetCDF settings (compression level, precision, MPI options)
        are properly configured for grid-based data output in CDF4 format.

        Raises
        ------
        ValueError
            If NetCDF compression level is out of range (0-9) or precision
            setting is invalid (must be 'single' or 'double')
        """
        logger.info("NetcdfOptions.validate()")

        # Validate compression level
        if not (0 <= self.cdf_compress <= 9):
            raise ValueError(
                f"Invalid CDF.Compress value: {self.cdf_compress}. Must be between 0 and 9."
            )

        # Validate grid precision
        valid_precisions = ["single", "double"]
        if self.cdf_grid_precision.lower() not in valid_precisions:
            raise ValueError(
                f"Invalid CDF.Grid.Precision: {self.cdf_grid_precision}. Must be 'single' or 'double'."
            )

    def update_from_fdf(self, fdf_dict: Dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in ["cdf.save", "cdf_save"]:
                self.cdf_save = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["cdf.compress", "cdf_compress"]:
                self.cdf_compress = int(value)
            elif key_lower in ["cdf.mpi", "cdf_mpi"]:
                self.cdf_mpi = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["cdf.grid.precision", "cdf_grid_precision"]:
                self.cdf_grid_precision = str(value)

    def generate_fdf(self) -> Dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns:
            Dictionary of FDF parameters
        """
        fdf: Dict[str, Any] = {}

        # CDF.Save - always write with default marker
        if not self.cdf_save:
            fdf["CDF.Save"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["CDF.Save"] = "true"

        # CDF.Compress - always write with default marker
        if self.cdf_compress == 0:
            fdf["CDF.Compress"] = "0  # SIESTA DEFAULT VALUE"
        else:
            fdf["CDF.Compress"] = str(self.cdf_compress)

        # CDF.MPI - always write with default marker
        if not self.cdf_mpi:
            fdf["CDF.MPI"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["CDF.MPI"] = "true"

        # CDF.Grid.Precision - always write with default marker
        if self.cdf_grid_precision.lower() == "single":
            fdf[
                "CDF.Grid.Precision"
            ] = f"{self.cdf_grid_precision}  # SIESTA DEFAULT VALUE"
        else:
            fdf["CDF.Grid.Precision"] = self.cdf_grid_precision

        return fdf

    def to_ase(self) -> Dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns:
            Dictionary of ASE parameters
        """
        # ASE doesn't have NetCDF output parameters
        # These are SIESTA-specific file format options
        return {}

    def generate_netcdf_block(self):
        """
        Generate FDF arguments for NetCDF output with comment header.

        Populates netcdf_fdf_arguments dictionary with all NetCDF parameters
        that are set to non-default values. Adds comment header if comments are enabled.
        """
        logger.info("NetcdfOptions.generate_netcdf_block()")

        # Collect parameters first (only non-default values)
        params_to_add = {}

        if self.cdf_save:  # False is default
            params_to_add["CDF.Save"] = self.cdf_save
        if self.cdf_compress != 0:  # 0 is default
            params_to_add["CDF.Compress"] = self.cdf_compress
        if self.cdf_mpi:  # False is default
            params_to_add["CDF.MPI"] = self.cdf_mpi
        if self.cdf_grid_precision.lower() != "single":  # 'single' is default
            params_to_add["CDF.Grid.Precision"] = self.cdf_grid_precision

        # Only add comment header if there are parameters to add
        if params_to_add:
            if self.comments:
                self.netcdf_fdf_arguments["#NetcdfOptions"] = self.comments
            self.netcdf_fdf_arguments.update(params_to_add)

    @classmethod
    def setup_netcdf_settings(
        cls, user_params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> "NetcdfOptions":
        """
        Create and configure a NetcdfOptions instance with full parameter parsing.

        This method handles proper key normalization, type conversion, and fuzzy matching
        to configure NetCDF output settings from user parameters. Supports SIESTA FDF
        parameter names (CDF.Save, CDF.Compress, etc.) with automatic conversion.

        Args:
            user_params: Dictionary of user-defined parameters (case-insensitive, may include dots).
                        If None or empty, all default values are used.
            **kwargs: Additional keyword arguments to override or supplement user_params.

        Returns:
            NetcdfOptions: Configured instance with all fields set.

        Examples:
            >>> # Using SIESTA FDF parameter names
            >>> netcdf = NetcdfOptions.setup_netcdf_settings({
            ...     "CDF.Save": True,
            ...     "CDF.Compress": 6,
            ...     "CDF.Grid.Precision": "double"
            ... })

            >>> # Using Python attribute names
            >>> netcdf = NetcdfOptions.setup_netcdf_settings({
            ...     "cdf_save": True,
            ...     "cdf_compress": 6,
            ...     "cdf_grid_precision": "double"
            ... })
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]NetcdfOptions.setup_netcdf_settings()[/green]")

        # Initialize instance with defaults
        instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default NetcdfOptions values.[/blue]"
                )
            return instance

        # Get valid attribute names (lowercase for comparison)
        netcdf_attributes = {
            field.name.lower()
            for field in fields(cls)
            if not field.name.startswith("_") and field.name != "CONSOLE_VERBOSITY"
        }
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                f"[blue]Available NetcdfOptions attributes: {netcdf_attributes}[/blue]"
            )

        # Process user parameters
        import re
        from difflib import get_close_matches

        for key, value in user_params.items():
            # Normalize key: handle camelCase and remove dots
            key_with_underscores = re.sub(r"([a-z])([A-Z])", r"\1_\2", key)
            key_normalized = key_with_underscores.replace(".", "_").lower()

            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Processing key: {key} -> {key_normalized}, value: {value}[/blue]"
                )

            # Check if normalized key matches any attribute
            matched_attr = None
            if key_normalized in netcdf_attributes:
                matched_attr = key_normalized
            else:
                # Try fuzzy matching
                close_matches = get_close_matches(
                    key_normalized, netcdf_attributes, n=1, cutoff=0.6
                )
                if close_matches:
                    matched_attr = close_matches[0]
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
                        console.print(
                            f"[yellow]Fuzzy match: '{key}' -> '{matched_attr}'[/yellow]"
                        )

            # Set attribute if matched
            if matched_attr:
                # Type conversion based on parameter type
                if matched_attr == "cdf_compress":
                    # Integer parameter with range validation
                    try:
                        compress_val = int(value)
                        if 0 <= compress_val <= 9:
                            setattr(instance, matched_attr, compress_val)
                        else:
                            console.print(
                                f"[yellow]Warning: CDF.Compress must be 0-9, got {compress_val}. Using default 0.[/yellow]"
                            )
                    except (ValueError, TypeError):
                        console.print(
                            f"[yellow]Warning: Could not convert '{value}' to int for '{matched_attr}'. Using default.[/yellow]"
                        )
                elif matched_attr == "cdf_grid_precision":
                    # String parameter with validation
                    valid_precisions = ["single", "double"]
                    if isinstance(value, str):
                        if value.lower() in valid_precisions:
                            setattr(instance, matched_attr, value.lower())
                        else:
                            console.print(
                                f"[yellow]Warning: '{value}' not in {valid_precisions}. Using default 'single'.[/yellow]"
                            )
                    else:
                        setattr(instance, matched_attr, str(value))
                elif matched_attr in ["cdf_save", "cdf_mpi"]:
                    # Boolean parameters
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
                else:
                    # Direct assignment for other types
                    setattr(instance, matched_attr, value)
            else:
                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
                    console.print(
                        f"[yellow]Warning: No match found for parameter '{key}' in NetcdfOptions[/yellow]"
                    )

        # Generate FDF block with comment header
        instance.generate_netcdf_block()

        return instance
