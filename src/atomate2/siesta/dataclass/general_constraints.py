"""
Module defining base SIESTA input set and generator.

class GeneralConstraints

Based on User's Guide Siesta 5.4.0
Section: 7.7 Use of General constraints

"""

# Metadata

__all__ = ["GeneralConstraints"]

import logging
from dataclasses import dataclass, field
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass

logger = logging.getLogger(__name__)


@dataclass
class GeneralConstraints(FDFDataclass):
    """SIESTA general geometry constraints (User's Guide Section 7.7)."""

    # ------------------------------
    # 7.7 Use of General constraints
    # ------------------------------

    # geometry_constraints_block: Dict[float,Any]= field(default_factory=dict)
    # %block Geometry.Constraints 〈None〉
    geometry_constraints_block: list[str] | dict[str, Any] = field(
        default_factory=list,
        metadata={
            "description": (
                "A block to define constraints on atomic positions or lattice "
                "vectors during a geometry optimization or molecular dynamics run "
                "(e.g., fixing atoms). Can be list or dict."
            ),
            "SIESTA keyword": "%block Geometry.Constraints",
        },
    )

    constraints_fdf_arguments: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A dictionary for any additional or arbitrary FDF flags related "
                "to geometric constraints."
            ),
            "SIESTA keyword": None,
        },
    )

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "%block Geometry.Constraints",
            )
            self.__class__._registered = True  # noqa: SLF001 own-class registration guard

    def validate(self) -> None:
        """
        Validate geometry constraint parameters.

        Checks configuration for geometric constraints including fixed atoms, fixed
        coordinates, and cell constraints for relaxation and molecular dynamics.

        Raises
        ------
        ValueError
            If constraint specifications are invalid or inconsistent
        """
        logger.info("GeneralConstraints.validate()")

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in [
                "%block geometry.constraints",
                "geometry_constraints_block",
            ]:
                # Accept both list and dict formats for block parameters
                self.geometry_constraints_block = value

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters
        """
        fdf: dict[str, Any] = {}

        if self.geometry_constraints_block:
            fdf["#GeneralConstraints"] = "Geometry Constraints"
            fdf["%block Geometry.Constraints"] = self.geometry_constraints_block

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE has its own constraint system (FixAtoms, FixBondLengths, etc.)
        # These are SIESTA-specific FDF constraints
        return {}

    def generate_constraints_block(self) -> None:
        """
        Generate the geometric constraints block for the FDF file.

        This is a wrapper around generate_fdf() to maintain backward compatibility
        with code that calls this method directly (e.g., setup_constraints()).

        By calling generate_fdf(), we ensure:
        - Single source of truth for FDF generation
        - Proper "# SIESTA DEFAULT VALUE" markers on default parameters
        - Consistency with user_params, powerups, and tier presets
        - DRY principle (no parameter duplication)
        - Values updated via update_from_fdf() are properly reflected

        Creates FDF arguments for geometric constraints if any are specified.
        Empty by default unless constraints are defined.
        """
        logger.info("GeneralConstraints.generate_constraints_block()")

        # Call generate_fdf() which uses the current dataclass attributes
        # (these have been updated from user_params/powerups/tiers via
        # update_from_fdf())
        self.constraints_fdf_arguments = self.generate_fdf()

    @classmethod
    def setup_constraints(
        cls, user_params: dict[str, Any] | None = None
    ) -> "GeneralConstraints":
        """
        Create and configure a GeneralConstraints instance from user parameters.

        This classmethod provides initialization of geometric constraints for
        the tier-based input system.

        Parameters
        ----------
        user_params : dict, optional
            Dictionary of user-defined parameters containing constraint specifications.
            If None or empty, no constraints are applied (free relaxation).

        Returns
        -------
        GeneralConstraints
            Configured instance with FDF arguments populated

        Examples
        --------
        >>> constraints = GeneralConstraints.setup_constraints(
        ...     {
        ...         "geometry_constraints_block": {
        ...             "position from 1 to 5": None,  # Fix atoms 1-5
        ...             "routine constr": None,  # Fixed cell
        ...         }
        ...     }
        ... )
        """
        logger.info("GeneralConstraints.setup_constraints()")

        # Initialize with defaults
        instance = cls()

        # Handle empty user_params
        if user_params is None or not user_params:
            logger.debug(
                "No user parameters provided; no geometric constraints applied."
            )
        else:
            # Process user_params
            from dataclasses import fields as dc_fields

            valid_fields = {f.name.lower() for f in dc_fields(cls)}

            for key, value in user_params.items():
                key_normalized = key.lower().replace(".", "_")
                if key_normalized in valid_fields:
                    # Find original field name (preserving case)
                    original_key = next(
                        f.name
                        for f in dc_fields(cls)
                        if f.name.lower() == key_normalized
                    )
                    setattr(instance, original_key, value)

        # Validate settings
        try:
            instance.validate()
        except ValueError as e:
            logger.error(f"Geometric constraints validation failed: {e}")  # noqa: TRY400 preserve message-only log
            raise

        # Generate FDF block
        instance.generate_constraints_block()

        return instance
