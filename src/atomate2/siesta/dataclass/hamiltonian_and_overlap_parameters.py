"""
Module defining Hamiltonian and overlap matrix parameters for SIESTA calculations.

This module provides configuration for matrix element handling including sparsity
approximations, debugging output, and matrix storage options. These are expert-level
parameters for controlling electronic structure calculation details.

class HamiltonianAndOverlapParameters

Based on User's Guide Siesta 5.4.0
Section: 6.11 Matrix elements of the Hamiltonian and overlap
         6.11.1 The auxiliary supercell
"""

# Metadata

__all__ = ["HamiltonianAndOverlapParameters"]

import logging
from dataclasses import dataclass, field, fields
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@dataclass
class HamiltonianAndOverlapParameters(FDFDataclass):
    """
    Configuration for Hamiltonian and overlap matrix elements in SIESTA.

    This class manages settings for Hamiltonian (H) and overlap (S) matrix
    element calculations, including sparsity approximations, debugging output,
    and matrix storage. These are expert-level parameters for optimizing large-scale
    calculations.

    Parameters
    ----------
    negl_non_overlap_int : bool
        Neglect integrals between non-overlapping orbitals (speedup for sparse
        systems). Default: False
    scf_write_extra : bool
        Write extra debugging info during SCF cycle. Default: False
    save_hs : bool
        Save Hamiltonian and overlap matrices to .HS file (needed for
        post-processing). Default: True
    force_aux_cell : bool
        Force use of auxiliary supercell for force calculations (charged
        systems). Default: False

    Methods
    -------
    validate()
        Validate Hamiltonian and overlap matrix configuration
    setup_hamiltonian_settings(user_params)
        Create configured instance with fuzzy parameter matching
    """

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = VerbosityLevel.ERROR

    # ---------------------------------------------------
    # 6.11 Matrix elements of the Hamiltonian and overlap
    # ---------------------------------------------------
    negl_non_overlap_int: bool = field(
        default=False,
        metadata={
            "description": "A flag to neglect integrals between basis orbitals "
            "with zero spatial overlap. This is an approximation that can speed "
            "up calculations for very large, sparse systems.",
            "SIESTA keyword": "Negl.NonOverlap.Int",
        },
    )

    scf_write_extra: bool = field(
        default=False,
        metadata={
            "description": "A debugging flag to write extra, detailed "
            "information to the main output file during the SCF cycle.",
            "SIESTA keyword": "SCF.Write.Extra",
        },
    )

    save_hs: bool = field(
        default=True,
        metadata={
            "description": "A flag to save the final converged Hamiltonian (H) "
            "and Overlap (S) matrices to a file (.HS), which is required for "
            "many post-processing tasks like band structure analysis.",
            "SIESTA keyword": "SaveHS",
        },
    )

    # ------------------------------
    # 6.11.1 The auxiliary supercell
    # ------------------------------
    force_aux_cell: bool = field(
        default=False,
        metadata={
            "description": "A flag to force the use of an auxiliary supercell "
            "for the calculation of forces, which can be necessary for charged "
            "systems or systems in an electric field.",
            "SIESTA keyword": "ForceAuxCell",
        },
    )

    # Comment header for FDF output
    comments: str = field(
        default="# Hamiltonian and Overlap Matrix Configuration "
        "(HamiltonianAndOverlapParameters dataclass module)",
        metadata={"description": "Comment header for FDF file"},
    )

    # Dictionary to hold FDF arguments
    hamiltonian_fdf_arguments: dict[str, Any] = field(default_factory=dict)

    # Track which parameters were explicitly provided by user
    _user_provided_params: set = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "Negl.NonOverlap.Int",
                "SCF.Write.Extra",
                "SaveHS",
                "ForceAuxCell",
            )
            self.__class__._registered = True  # noqa: SLF001 class-level registration guard

    def validate(self) -> None:
        """
        Validate Hamiltonian and overlap matrix parameters.

        Checks configuration for matrix element handling including sparsity
        approximations and matrix storage options.

        Raises
        ------
        ValueError
            If parameters are inconsistent (currently no validation rules defined)
        """
        logger.info("HamiltonianAndOverlapParameters.validate()")

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in ["negl.nonoverlap.int", "negl_non_overlap_int"]:
                self.negl_non_overlap_int = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["scf.write.extra", "scf_write_extra"]:
                self.scf_write_extra = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["savehs", "save_hs"]:
                self.save_hs = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["forceauxcell", "force_aux_cell"]:
                self.force_aux_cell = (
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
        fdf["#HamiltonianAndOverlap"] = "Hamiltonian and Overlap Settings"

        # Negl.NonOverlap.Int - always write with default marker
        if not self.negl_non_overlap_int:
            fdf["Negl.NonOverlap.Int"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["Negl.NonOverlap.Int"] = "true"

        # SCF.Write.Extra - always write with default marker
        if not self.scf_write_extra:
            fdf["SCF.Write.Extra"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["SCF.Write.Extra"] = "true"

        # SaveHS - always write with default marker
        if self.save_hs:
            fdf["SaveHS"] = "true  # SIESTA DEFAULT VALUE"
        else:
            fdf["SaveHS"] = "false"

        # ForceAuxCell - always write with default marker
        if not self.force_aux_cell:
            fdf["ForceAuxCell"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["ForceAuxCell"] = "true"

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have direct equivalents for these Hamiltonian/overlap parameters
        return {}

    def generate_hamiltonian_block(self) -> None:
        """
        Generate FDF arguments for Hamiltonian and overlap with comment header.

        Populates hamiltonian_fdf_arguments dictionary with all matrix parameters
        that are set to non-default values OR were explicitly provided by user.
        Adds comment header if comments are enabled.
        """
        logger.info("HamiltonianAndOverlapParameters.generate_hamiltonian_block()")

        # Collect parameters first (non-default values OR explicitly provided)
        params_to_add = {}

        # Check if explicitly provided OR differs from default
        if (
            self.negl_non_overlap_int
            or "negl_non_overlap_int" in self._user_provided_params
        ):  # False is default
            params_to_add["Negl.NonOverlap.Int"] = self.negl_non_overlap_int
        if (
            self.scf_write_extra or "scf_write_extra" in self._user_provided_params
        ):  # False is default
            params_to_add["SCF.Write.Extra"] = self.scf_write_extra
        if (
            not self.save_hs or "save_hs" in self._user_provided_params
        ):  # True is default
            params_to_add["SaveHS"] = self.save_hs
        if (
            self.force_aux_cell or "force_aux_cell" in self._user_provided_params
        ):  # False is default
            params_to_add["ForceAuxCell"] = self.force_aux_cell

        # Only add comment header if there are parameters to add
        if params_to_add:
            if self.comments:
                self.hamiltonian_fdf_arguments["#HamiltonianAndOverlapParameters"] = (
                    self.comments
                )
            self.hamiltonian_fdf_arguments.update(params_to_add)

    @classmethod
    def setup_hamiltonian_settings(
        cls,
        user_params: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG003
    ) -> "HamiltonianAndOverlapParameters":
        """
        Create and configure a HamiltonianAndOverlapParameters instance.

        This method handles proper key normalization, type conversion, and
        fuzzy matching to configure Hamiltonian and overlap matrix settings
        from user parameters. Supports SIESTA FDF parameter names (SaveHS,
        ForceAuxCell, etc.) with automatic conversion.

        Args:
            user_params: Dictionary of user-defined parameters (case-insensitive,
                may include dots). If None or empty, all default values are used.
            **kwargs: Additional keyword arguments to override or supplement
                user_params.

        Returns
        -------
            HamiltonianAndOverlapParameters: Configured instance with all fields set.

        Examples
        --------
            >>> # Using SIESTA FDF parameter names
            >>> hamiltonian = (
            ...     HamiltonianAndOverlapParameters.setup_hamiltonian_settings(
            ...         {
            ...             "SaveHS": True,
            ...             "Negl.NonOverlap.Int": False,
            ...             "ForceAuxCell": True,
            ...         }
            ...     )
            ... )

            >>> # Using Python attribute names
            >>> hamiltonian = (
            ...     HamiltonianAndOverlapParameters.setup_hamiltonian_settings(
            ...         {
            ...             "save_hs": True,
            ...             "negl_non_overlap_int": False,
            ...             "force_aux_cell": True,
            ...         }
            ...     )
            ... )
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]HamiltonianAndOverlapParameters.setup_hamiltonian_settings()[/green]"
            )

        # Initialize instance with defaults
        instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default "
                    "HamiltonianAndOverlapParameters values.[/blue]"
                )
            return instance

        # Get valid attribute names (lowercase for comparison)
        hamiltonian_attributes = {
            field.name.lower()
            for field in fields(cls)
            if not field.name.startswith("_") and field.name != "CONSOLE_VERBOSITY"
        }
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                f"[blue]Available HamiltonianAndOverlapParameters attributes: "
                f"{hamiltonian_attributes}[/blue]"
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
            if key_normalized in hamiltonian_attributes:
                matched_attr = key_normalized
            else:
                # Try fuzzy matching
                close_matches = get_close_matches(
                    key_normalized, hamiltonian_attributes, n=1, cutoff=0.6
                )
                if close_matches:
                    matched_attr = close_matches[0]
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
                        console.print(
                            f"[yellow]Fuzzy match: '{key}' -> '{matched_attr}'[/yellow]"
                        )

            # Set attribute if matched
            if matched_attr:
                # Track that this parameter was explicitly provided
                instance._user_provided_params.add(matched_attr)

                # All parameters are boolean for this module
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
            elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
                console.print(
                    f"[yellow]Warning: No match found for parameter '{key}' "
                    f"in HamiltonianAndOverlapParameters[/yellow]"
                )

        # Generate FDF block with comment header
        instance.generate_hamiltonian_block()

        return instance
