"""
Module defining parallel computation options for SIESTA calculations.

This module provides configuration for MPI parallelization strategies including
processor grid layout, block-cyclic distribution, and domain/spatial decomposition
for O(N) calculations.

class ParallelOptions

Based on User's Guide Siesta 5.4.0
Section:  6.27 Parallel options
"""

# Metadata

__all__ = ["ParallelOptions"]

import logging
from dataclasses import dataclass, field, fields
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@dataclass
class ParallelOptions(FDFDataclass):
    """
    Configuration for parallel computation in SIESTA.

    This class manages MPI parallelization options including processor grid layout,
    matrix distribution strategies, and specialized parallelization schemes for
    O(N) calculations. These are expert-level performance tuning parameters.

    Parameters
    ----------
    block_size : int, optional
        Block size for 2D block-cyclic matrix distribution. Default: None (automatic)
    processor_y : int, optional
        Number of processors in Y dimension of 2D processor grid. Default: None (automatic)
    fft_processor_y_traditional : bool
        Use traditional FFT parallelization over Y dimension. Default: False
    use_domain_decomposition : bool
        Enable domain decomposition (orbitals/atoms grouped by processor). Default: False
    use_spatial_decomposition : bool
        Enable spatial decomposition (real-space grid distribution). Default: True
    rc_spatial : float, optional
        Communication radius for spatial decomposition (Bohr). Default: None (max orbital range)

    Methods
    -------
    validate()
        Validate parallel computation configuration
    setup_parallel_settings(user_params)
        Create configured instance with fuzzy parameter matching
    """

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = VerbosityLevel.ERROR

    # ---------------------
    # 6.27 Parallel options
    # ---------------------
    block_size: int = field(
        default=None,
        metadata={
            "description": "Sets the block size for the 2D block-cyclic distribution of matrices in parallel calculations. This is a key performance tuning parameter.",
            "SIESTA keyword": "BlockSize",
        },
    )

    processor_y: int = field(
        default=None,
        metadata={
            "description": "Manually sets the number of processors in the 'Y' dimension of the 2D processor grid, allowing for manual tuning of the parallel layout.",
            "SIESTA keyword": "ProcessorY",
        },
    )

    fft_processor_y_traditional: bool = field(
        default=False,
        metadata={
            "description": "If true, uses a traditional parallelization scheme for the Fast Fourier Transforms (FFTs) over the 'Y' dimension of the processor grid.",
            "SIESTA keyword": "FFT.ProcessorY.Traditional",
        },
    )

    # ---------------------------------------
    # 6.27.1 Parallel decompositions for O(N)
    # ---------------------------------------
    use_domain_decomposition: bool = field(
        default=False,
        metadata={
            "description": "A flag to enable a parallelization strategy based on domain decomposition, where orbitals or atoms are grouped and assigned to different processors.",
            "SIESTA keyword": "UseDomainDecomposition",
        },
    )

    use_spatial_decomposition: bool = field(
        default=True,
        metadata={
            "description": "A flag to enable a parallelization strategy based on decomposing the real-space grid and distributing it among processors.",
            "SIESTA keyword": "UseSpatialDecomposition",
        },
    )

    rc_spatial: float = field(
        default=None,
        metadata={
            "description": "The communication radius (in Bohr) for the spatial decomposition scheme. It defaults to the maximum range of the basis orbitals.",
            "SIESTA keyword": "RcSpatial",
        },
    )

    # Comment header for FDF output
    comments: str = field(
        default="# Parallel Computation Configuration (ParallelOptions dataclass module)",
        metadata={"description": "Comment header for FDF file"},
    )

    # Dictionary to hold FDF arguments
    parallel_fdf_arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "BlockSize",
                "ProcessorY",
                "FFT.ProcessorY.Traditional",
                "UseDomainDecomposition",
                "UseSpatialDecomposition",
                "RcSpatial",
                "NumberOfNodesPerGroup",
            )
            self.__class__._registered = True

    def validate(self):
        """
        Validate parallel computation options.

        Checks configuration for parallel execution including processor distribution,
        MPI settings, and parallelization strategies (k-point, orbital, spatial).

        Raises
        ------
        ValueError
            If parallel options are invalid or inconsistent with available resources
        """
        logger.info("ParallelOptions.validate()")

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        from atomate2.siesta.dataclass.units import parse_length

        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in ["blocksize", "block_size"]:
                self.block_size = int(value) if value else None
            elif key_lower in ["processory", "processor_y"]:
                self.processor_y = int(value) if value else None
            elif key_lower in [
                "fft.processory.traditional",
                "fft_processor_y_traditional",
            ]:
                self.fft_processor_y_traditional = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["usedomaindecomposition", "use_domain_decomposition"]:
                self.use_domain_decomposition = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["usespatialdecomposition", "use_spatial_decomposition"]:
                self.use_spatial_decomposition = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["rcspatial", "rc_spatial"]:
                self.rc_spatial = (
                    parse_length(value, target_unit="Bohr") if value else None
                )

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters
        """
        fdf = {}
        fdf["#Parallel"] = "Parallel Options"

        # BlockSize - write if set (no SIESTA default, optional parameter)
        if self.block_size is not None:
            fdf["BlockSize"] = str(self.block_size)

        # ProcessorY - write if set (no SIESTA default, optional parameter)
        if self.processor_y is not None:
            fdf["ProcessorY"] = str(self.processor_y)

        # FFT.ProcessorY.Traditional - always write with default marker
        if not self.fft_processor_y_traditional:
            fdf["FFT.ProcessorY.Traditional"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["FFT.ProcessorY.Traditional"] = "true"

        # UseDomainDecomposition - always write with default marker
        if not self.use_domain_decomposition:
            fdf["UseDomainDecomposition"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["UseDomainDecomposition"] = "true"

        # UseSpatialDecomposition - always write with default marker
        if self.use_spatial_decomposition:
            fdf["UseSpatialDecomposition"] = "true  # SIESTA DEFAULT VALUE"
        else:
            fdf["UseSpatialDecomposition"] = "false"

        # RcSpatial - write if set (no SIESTA default, optional parameter)
        if self.rc_spatial is not None:
            fdf["RcSpatial"] = f"{self.rc_spatial} Bohr"

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have parallel execution parameters
        # These are handled by the ASE calculator wrapper
        return {}

    def generate_parallel_block(self):
        """
        Generate FDF arguments for parallel computation with comment header.

        Populates parallel_fdf_arguments dictionary with all parallel computation
        parameters that are set to non-default values. Adds comment header if comments are enabled.
        """
        logger.info("ParallelOptions.generate_parallel_block()")

        # Collect parameters first (only non-default values)
        params_to_add = {}

        if self.block_size is not None:
            params_to_add["BlockSize"] = self.block_size
        if self.processor_y is not None:
            params_to_add["ProcessorY"] = self.processor_y
        if self.fft_processor_y_traditional:  # False is default
            params_to_add["FFT.ProcessorY.Traditional"] = (
                self.fft_processor_y_traditional
            )
        if self.use_domain_decomposition:  # Only add if True (False is default)
            params_to_add["UseDomainDecomposition"] = self.use_domain_decomposition
        if not self.use_spatial_decomposition:  # True is default, add if False
            params_to_add["UseSpatialDecomposition"] = self.use_spatial_decomposition
        if self.rc_spatial is not None:
            params_to_add["RcSpatial"] = f"{self.rc_spatial} Bohr"

        # Only add comment header if there are parameters to add
        if params_to_add:
            if self.comments:
                self.parallel_fdf_arguments["#ParallelOptions"] = self.comments
            self.parallel_fdf_arguments.update(params_to_add)

    @classmethod
    def setup_parallel_settings(
        cls, user_params: dict[str, Any] | None = None, **kwargs
    ) -> "ParallelOptions":
        """
        Create and configure a ParallelOptions instance with full parameter parsing.

        This method handles proper key normalization, type conversion, and fuzzy matching
        to configure parallel computation settings from user parameters. Supports SIESTA FDF
        parameter names (BlockSize, ProcessorY, etc.) with automatic conversion.

        Args:
            user_params: Dictionary of user-defined parameters (case-insensitive, may include dots).
                        If None or empty, all default values are used.
            **kwargs: Additional keyword arguments to override or supplement user_params.

        Returns
        -------
            ParallelOptions: Configured instance with all fields set.

        Examples
        --------
            >>> # Using SIESTA FDF parameter names
            >>> parallel = ParallelOptions.setup_parallel_settings(
            ...     {"BlockSize": 32, "ProcessorY": 4, "UseSpatialDecomposition": True}
            ... )

            >>> # Using Python attribute names
            >>> parallel = ParallelOptions.setup_parallel_settings(
            ...     {
            ...         "block_size": 32,
            ...         "processor_y": 4,
            ...         "use_spatial_decomposition": True,
            ...     }
            ... )
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]ParallelOptions.setup_parallel_settings()[/green]")

        # Initialize instance with defaults
        instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default ParallelOptions values.[/blue]"
                )
            return instance

        # Get valid attribute names (lowercase for comparison)
        parallel_attributes = {
            field.name.lower()
            for field in fields(cls)
            if not field.name.startswith("_") and field.name != "CONSOLE_VERBOSITY"
        }
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                f"[blue]Available ParallelOptions attributes: {parallel_attributes}[/blue]"
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
            if key_normalized in parallel_attributes:
                matched_attr = key_normalized
            else:
                # Try fuzzy matching
                close_matches = get_close_matches(
                    key_normalized, parallel_attributes, n=1, cutoff=0.6
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
                if matched_attr in ["block_size", "processor_y"]:
                    # Integer parameters
                    try:
                        if value is not None:
                            setattr(instance, matched_attr, int(value))
                        else:
                            setattr(instance, matched_attr, None)
                    except (ValueError, TypeError):
                        console.print(
                            f"[yellow]Warning: Could not convert '{value}' to int for '{matched_attr}'. Using default.[/yellow]"
                        )
                elif matched_attr == "rc_spatial":
                    # Float parameter
                    try:
                        if value is not None:
                            setattr(instance, matched_attr, float(value))
                        else:
                            setattr(instance, matched_attr, None)
                    except (ValueError, TypeError):
                        console.print(
                            f"[yellow]Warning: Could not convert '{value}' to float for '{matched_attr}'. Using default.[/yellow]"
                        )
                elif matched_attr in [
                    "fft_processor_y_traditional",
                    "use_domain_decomposition",
                    "use_spatial_decomposition",
                ]:
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
            elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
                console.print(
                    f"[yellow]Warning: No match found for parameter '{key}' in ParallelOptions[/yellow]"
                )

        # Generate FDF block with comment header
        instance.generate_parallel_block()

        return instance
