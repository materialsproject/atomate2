"""
Module defining efficiency and performance monitoring options for SIESTA calculations.

This module provides configuration for memory reporting, CPU/walltime accounting,
timing analysis, and restart data management. These are expert-level optimization
and monitoring parameters.

class EfficiencyOptions

Based on User's Guide Siesta 5.4.0
Section: 6.28 Efficiency options
Section: 6.29 Memory, CPU-time, and Wall time accounting options
Section: 6.30 The catch-all option UseSaveData
"""

# Metadata

__all__ = ["EfficiencyOptions"]

import logging
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyOptions(FDFDataclass):
    """
    Configuration for efficiency and performance monitoring in SIESTA.

    This class manages settings for memory allocation reporting, timing analysis,
    walltime limits, and restart data usage. These are expert-level parameters for
    performance optimization and job management.

    Parameters
    ----------
    direct_phi : bool
        Use direct real-space summation for electrostatic potential (slower).
        Default: False
    alloc_report_level : int
        Verbosity level for memory allocation report (0=minimal). Default: 0
    alloc_report_threshold : float
        Minimum memory allocation size to report (Mbytes). Default: 0.0
    timer_report_threshold : float
        Minimum time to include routine in timing report (seconds). Default: 0.0
    user_tree_timer : bool
        Use tree-like structure for timing reports. Default: False
    user_parallel_timer : bool
        Synchronize timers across parallel processors. Default: True
    timing_split_scf_steps : bool
        Separate timing info for each SCF step. Default: False
    max_walltime : float, optional
        Maximum wall-clock time for job (seconds). Default: None (infinite)
    max_walltime_slack : float
        Time before max_walltime to initiate clean shutdown (seconds). Default: 5.0
    use_save_data : bool
        Enable use of restart data (.DM, .XV files). Default: False

    Methods
    -------
    validate()
        Validate efficiency and performance configuration
    setup_efficiency_settings(user_params)
        Create configured instance with fuzzy parameter matching
    """

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = VerbosityLevel.ERROR

    # -----------------------
    # 6.28 Efficiency options
    # -----------------------
    direct_phi: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, computes the electrostatic potential via a direct "
                "real-space summation instead of using Fast Fourier Transforms "
                "(FFTs). This is slower but can be used for open boundary "
                "conditions."
            ),
            "SIESTA keyword": "DirectPhi",
        },
    )

    # -------------------------------------------------------
    # 6.29 Memory, CPU-time, and Wall time accounting options
    # -------------------------------------------------------
    alloc_report_level: int = field(
        default=0,
        metadata={
            "description": (
                "Sets the verbosity level for the memory allocation report at "
                "the end of the run."
            ),
            "SIESTA keyword": "AllocReportLevel",
        },
    )

    alloc_report_threshold: float = field(
        default=0.0,
        metadata={
            "description": (
                "The minimum memory allocation size (in Mbytes) to be included "
                "in the allocation report."
            ),
            "SIESTA keyword": "AllocReportThreshold",
        },
    )

    timer_report_threshold: float = field(
        default=0.0,
        metadata={
            "description": (
                "The minimum time (in seconds) for a routine to be included in "
                "the timing report."
            ),
            "SIESTA keyword": "TimerReportThreshold",
        },
    )

    user_tree_timer: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, uses a tree-like structure for the timing reports, "
                "showing the hierarchy of routines."
            ),
            "SIESTA keyword": "UseTreeTimer",
        },
    )

    user_parallel_timer: bool = field(
        default=True,
        metadata={
            "description": (
                "If true, timers are synchronized across parallel processors to "
                "provide more accurate parallel timing information."
            ),
            "SIESTA keyword": "UseParallelTimer",
        },
    )

    timing_split_scf_steps: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, provides separate timing information for each "
                "individual SCF step."
            ),
            "SIESTA keyword": "TimingSplitScfSteps",
        },
    )

    max_walltime: float = field(
        default=None,
        metadata={
            "description": (
                "The maximum wall-clock time (in seconds) for the job. SIESTA "
                "will attempt a clean stop before this time is reached."
            ),
            "SIESTA keyword": "MaxWalltime",
        },
    )

    max_walltime_slack: float = field(
        default=5.0,
        metadata={
            "description": (
                "The slack time (in seconds) before the 'MaxWalltime' is "
                "reached, during which SIESTA will initiate a clean shutdown."
            ),
            "SIESTA keyword": "MaxWalltime.Slack",
        },
    )

    # -------------------------------------
    # 6.30 The catch-all option UseSaveData
    # -------------------------------------
    use_save_data: bool = field(
        default=False,
        metadata={
            "description": (
                "A global flag to enable the use of any available restart data, "
                "such as the density matrix (.DM) or atomic positions (.XV), to "
                "continue a previous calculation."
            ),
            "SIESTA keyword": "UseSaveData",
        },
    )

    # Comment header for FDF output
    comments: str = field(
        default=(
            "# Efficiency and Performance Configuration "
            "(EfficiencyOptions dataclass module)"
        ),
        metadata={"description": "Comment header for FDF file"},
    )

    # Dictionary to hold FDF arguments
    efficiency_fdf_arguments: dict[str, Any] = field(default_factory=dict)

    _registered: ClassVar[bool]

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "DirectPhi",
                "AllocReportLevel",
                "AllocReportThreshold",
                "TimerReportThreshold",
                "UseTreeTimer",
                "UserTreeTimer",
                "UseParallelTimer",
                "UserParallelTimer",
                "Timing.SplitSCFSteps",
                "TimingSplitSCFSteps",
                "MaxWalltime",
                "MaxWalltime.Slack",
                "MaxWalltimeSlack",
                "UseSaveData",
                "MD.UseSaveXV",
            )
            self.__class__._registered = True  # noqa: SLF001 class-level registration guard

    def validate(self) -> None:
        """
        Validate efficiency and performance options.

        Checks settings for memory reporting, CPU/walltime accounting, and restart
        data usage. Ensures walltime limits and reporting thresholds are properly
        configured.

        Raises
        ------
        ValueError
            If walltime parameters are invalid or reporting thresholds are negative
        """
        logger.info("EfficiencyOptions.validate()")

        # Validate thresholds are non-negative
        if self.alloc_report_threshold < 0:
            raise ValueError(
                "AllocReportThreshold must be non-negative, got "
                f"{self.alloc_report_threshold}"
            )
        if self.timer_report_threshold < 0:
            raise ValueError(
                "TimerReportThreshold must be non-negative, got "
                f"{self.timer_report_threshold}"
            )

        # Validate walltime parameters
        if self.max_walltime is not None and self.max_walltime <= 0:
            raise ValueError(f"MaxWalltime must be positive, got {self.max_walltime}")
        if self.max_walltime_slack < 0:
            raise ValueError(
                f"MaxWalltime.Slack must be non-negative, got {self.max_walltime_slack}"
            )

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in ["directphi", "direct_phi"]:
                self.direct_phi = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["allocreportlevel", "alloc_report_level"]:
                self.alloc_report_level = int(value)
            elif key_lower in ["allocreportthreshold", "alloc_report_threshold"]:
                self.alloc_report_threshold = float(value)
            elif key_lower in ["timerreportthreshold", "timer_report_threshold"]:
                self.timer_report_threshold = float(value)
            elif key_lower in ["usetreetimer", "usertreetimer", "user_tree_timer"]:
                self.user_tree_timer = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in [
                "useparalleltimer",
                "userparalleltimer",
                "user_parallel_timer",
            ]:
                self.user_parallel_timer = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in [
                "timing.splitscfsteps",
                "timingsplitscfsteps",
                "timing_split_scf_steps",
            ]:
                self.timing_split_scf_steps = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["maxwalltime", "max_walltime"]:
                self.max_walltime = float(value) if value else None
            elif key_lower in [
                "maxwalltime.slack",
                "maxwalltimeslack",
                "max_walltime_slack",
            ]:
                self.max_walltime_slack = float(value)
            elif key_lower in ["usesavedata", "use_save_data"]:
                self.use_save_data = (
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
        fdf = {}
        fdf["#Efficiency"] = "Efficiency and Performance Settings"

        # DirectPhi - always write with default marker
        if not self.direct_phi:
            fdf["DirectPhi"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["DirectPhi"] = "true"

        # AllocReportLevel - always write with default marker
        if self.alloc_report_level == 0:
            fdf["AllocReportLevel"] = "0  # SIESTA DEFAULT VALUE"
        else:
            fdf["AllocReportLevel"] = str(self.alloc_report_level)

        # AllocReportThreshold - always write with default marker
        if self.alloc_report_threshold == 0.0:
            fdf["AllocReportThreshold"] = "0.0 MB  # SIESTA DEFAULT VALUE"
        else:
            fdf["AllocReportThreshold"] = f"{self.alloc_report_threshold} MB"

        # TimerReportThreshold - always write with default marker
        if self.timer_report_threshold == 0.0:
            fdf["TimerReportThreshold"] = "0.0 s  # SIESTA DEFAULT VALUE"
        else:
            fdf["TimerReportThreshold"] = f"{self.timer_report_threshold} s"

        # UserTreeTimer - always write with default marker
        if not self.user_tree_timer:
            fdf["UseTreeTimer"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["UseTreeTimer"] = "true"

        # UserParallelTimer - always write with default marker
        if self.user_parallel_timer:
            fdf["UseParallelTimer"] = "true  # SIESTA DEFAULT VALUE"
        else:
            fdf["UseParallelTimer"] = "false"

        # TimingSplitScfSteps - always write with default marker
        if not self.timing_split_scf_steps:
            fdf["Timing.SplitSCFSteps"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["Timing.SplitSCFSteps"] = "true"

        # MaxWalltime - write if set (no default value in SIESTA)
        if self.max_walltime is not None:
            fdf["MaxWalltime"] = f"{self.max_walltime} s"

        # MaxWalltime.Slack - always write with default marker
        if self.max_walltime_slack == 5.0:
            fdf["MaxWalltime.Slack"] = "5.0 s  # SIESTA DEFAULT VALUE"
        else:
            fdf["MaxWalltime.Slack"] = f"{self.max_walltime_slack} s"

        # UseSaveData - always write with default marker
        if not self.use_save_data:
            fdf["UseSaveData"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["UseSaveData"] = "true"

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have efficiency/timing parameters
        # These are SIESTA-specific monitoring options
        return {}

    def generate_efficiency_block(self) -> None:
        """
        Generate FDF arguments for efficiency and performance with comment header.

        Populates efficiency_fdf_arguments dictionary with all efficiency parameters
        that are set to non-default values. Adds comment header if comments are enabled.
        """
        logger.info("EfficiencyOptions.generate_efficiency_block()")

        # Collect parameters first (only non-default values)
        params_to_add: dict[str, Any] = {}

        if self.direct_phi:  # False is default
            params_to_add["DirectPhi"] = self.direct_phi

        # Memory and timing reporting
        if self.alloc_report_level != 0:
            params_to_add["AllocReportLevel"] = self.alloc_report_level
        if self.alloc_report_threshold != 0.0:
            params_to_add["AllocReportThreshold"] = (
                f"{self.alloc_report_threshold} Mbytes"
            )
        if self.timer_report_threshold != 0.0:
            params_to_add["TimerReportThreshold"] = f"{self.timer_report_threshold} s"

        # Timer options
        if self.user_tree_timer:  # False is default
            params_to_add["UseTreeTimer"] = self.user_tree_timer
        if not self.user_parallel_timer:  # True is default
            params_to_add["UseParallelTimer"] = self.user_parallel_timer
        if self.timing_split_scf_steps:  # False is default
            params_to_add["TimingSplitScfSteps"] = self.timing_split_scf_steps

        # Walltime limits
        if self.max_walltime is not None:
            params_to_add["MaxWalltime"] = f"{self.max_walltime} s"
        if self.max_walltime_slack != 5.0:  # 5.0 is default
            params_to_add["MaxWalltime.Slack"] = f"{self.max_walltime_slack} s"

        # Restart data
        if self.use_save_data:  # False is default
            params_to_add["UseSaveData"] = self.use_save_data

        # Only add comment header if there are parameters to add
        if params_to_add:
            if self.comments:
                self.efficiency_fdf_arguments["#EfficiencyOptions"] = self.comments
            self.efficiency_fdf_arguments.update(params_to_add)

    @classmethod
    def setup_efficiency_settings(
        cls,
        user_params: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG003 interface kwarg
    ) -> "EfficiencyOptions":
        """
        Create and configure a EfficiencyOptions instance with full parameter parsing.

        This method handles proper key normalization, type conversion, and fuzzy
        matching to configure efficiency and performance settings from user
        parameters. Supports SIESTA FDF parameter names (DirectPhi,
        AllocReportLevel, etc.) with automatic conversion.

        Args:
            user_params: Dictionary of user-defined parameters (case-insensitive,
                        may include dots).
                        If None or empty, all default values are used.
            **kwargs: Additional keyword arguments to override or supplement
                        user_params.

        Returns
        -------
            EfficiencyOptions: Configured instance with all fields set.

        Examples
        --------
            >>> # Using SIESTA FDF parameter names
            >>> efficiency = EfficiencyOptions.setup_efficiency_settings(
            ...     {
            ...         "AllocReportLevel": 2,
            ...         "MaxWalltime": 86400,  # 24 hours
            ...         "UseSaveData": True,
            ...     }
            ... )

            >>> # Using Python attribute names
            >>> efficiency = EfficiencyOptions.setup_efficiency_settings(
            ...     {
            ...         "alloc_report_level": 2,
            ...         "max_walltime": 86400,
            ...         "use_save_data": True,
            ...     }
            ... )
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]EfficiencyOptions.setup_efficiency_settings()[/green]"
            )

        # Initialize instance with defaults
        instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default "
                    "EfficiencyOptions values.[/blue]"
                )
            return instance

        # Get valid attribute names (lowercase for comparison)
        efficiency_attributes = {
            field.name.lower()
            for field in fields(cls)
            if not field.name.startswith("_") and field.name != "CONSOLE_VERBOSITY"
        }
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                f"[blue]Available EfficiencyOptions attributes: "
                f"{efficiency_attributes}[/blue]"
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
                    f"[blue]Processing key: {key} -> {key_normalized}, "
                    f"value: {value}[/blue]"
                )

            # Check if normalized key matches any attribute
            matched_attr = None
            if key_normalized in efficiency_attributes:
                matched_attr = key_normalized
            else:
                # Try fuzzy matching
                close_matches = get_close_matches(
                    key_normalized, efficiency_attributes, n=1, cutoff=0.6
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
                if matched_attr == "alloc_report_level":
                    # Integer parameter
                    try:
                        setattr(instance, matched_attr, int(value))
                    except (ValueError, TypeError):
                        console.print(
                            f"[yellow]Warning: Could not convert '{value}' to int "
                            f"for '{matched_attr}'. Using default.[/yellow]"
                        )
                elif matched_attr in [
                    "alloc_report_threshold",
                    "timer_report_threshold",
                    "max_walltime",
                    "max_walltime_slack",
                ]:
                    # Float parameters
                    try:
                        if value is not None:
                            setattr(instance, matched_attr, float(value))
                        else:
                            setattr(instance, matched_attr, None)
                    except (ValueError, TypeError):
                        console.print(
                            f"[yellow]Warning: Could not convert '{value}' to "
                            f"float for '{matched_attr}'. Using default.[/yellow]"
                        )
                elif matched_attr in [
                    "direct_phi",
                    "user_tree_timer",
                    "user_parallel_timer",
                    "timing_split_scf_steps",
                    "use_save_data",
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
                    f"[yellow]Warning: No match found for parameter '{key}' "
                    f"in EfficiencyOptions[/yellow]"
                )

        # Generate FDF block with comment header
        instance.generate_efficiency_block()

        return instance
