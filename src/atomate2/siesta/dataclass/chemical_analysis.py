"""
Data class to manage chemical analysis options for SIESTA input.

class ChemicalAnalysis

Based on User's Guide Siesta 5.4.0
Section: 6.19 Options for chemical analysis
         6.19.2 Deprecated population flags
         6.19.3 Crystal-Orbitla overlap and hamiltonian populations (COOP/COHP)
"""

# Metadata

__all__ = ["ChemicalAnalysis"]

import logging
from dataclasses import dataclass, field
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass

logger = logging.getLogger(__name__)


@dataclass
class ChemicalAnalysis(FDFDataclass):
    """
    Data class to manage chemical analysis options for SIESTA input.

    Parameters
    ----------
    charge_mulliken : bool
        Enable modern Mulliken charge analysis. Default: False
    charge_hirshfeld : bool
        Enable Hirshfeld population analysis. Default: False
    charge_voronoi : bool
        Enable Voronoi population analysis. Default: False
    coop_write : bool
        Write Crystal Orbital Overlap Population analysis. Default: False

    Note: Many deprecated population flags are already available in
    DensityOfStatesAndBandStructure dataclass:
        - WriteMullikenPop, MullikenInSCF, SpinInSCF
        - Write.HirshfeldPop, Write.VoronoiPop
        - PartialChargesAtEveryGeometry, PartialChargesAtEverySCFStep
    """

    # -----------------------------------
    # 6.19 Options for chemical analysis
    # -----------------------------------

    charge_mulliken: str = field(
        default="none",
        metadata={
            "description": "Enable modern Mulliken charge analysis",
            "SIESTA keyword": "Charge.Mulliken",
        },
    )

    charge_mulliken_format: int = field(
        default=0,
        metadata={
            "description": "Enable modern Mulliken charge analysis",
            "SIESTA keyword": "Charge.Mulliken.Format",
        },
    )

    charge_hirshfeld: str = field(
        default="none",
        metadata={
            "description": "Enable Hirshfeld population analysis",
            "SIESTA keyword": "Charge.Hirshfeld",
        },
    )

    charge_voronoi: str = field(
        default="none",
        metadata={
            "description": "Enable Voronoi population analysis",
            "SIESTA keyword": "Charge.Voronoi",
        },
    )

    # -----------------------------------
    # 6.19.2 Deprecated population flags
    # -----------------------------------
    # TODO: 6.19.2 Deprecated population flags

    # -----------------------------------------------------------------------
    # 6.19.3 Crystal-Orbitla overlap and hamiltonian populations (COOP/COHP)
    # -----------------------------------------------------------------------

    coop_write: bool = field(
        default=False,
        metadata={
            "description": "Write Crystal Orbital Overlap Population analysis",
            "SIESTA keyword": "COOP.Write",
        },
    )
    # TODO: WFS.EnergyMin
    # TODO: WFS.EnergyMax

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "Charge.Mulliken",
                "Charge.Hirshfeld",
                "Charge.Voronoi",
                "COOP.Write",
            )
            self.__class__._registered = True  # noqa: SLF001 own-class registration guard

    def validate(self) -> None:
        """Validate the chemical analysis options block for the FDF file."""
        logger.info("ChemicalAnalysis.validate()")

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in ["charge.mulliken", "charge_mulliken"]:
                self.charge_mulliken = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["charge.hirshfeld", "charge_hirshfeld"]:
                self.charge_hirshfeld = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["charge.voronoi", "charge_voronoi"]:
                self.charge_voronoi = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["coop.write", "coop_write"]:
                self.coop_write = (
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

        # Charge.Mulliken - always write with default marker
        if not self.charge_mulliken:
            fdf["Charge.Mulliken"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["Charge.Mulliken"] = "true"

        # Charge.Hirshfeld - always write with default marker
        if not self.charge_hirshfeld:
            fdf["Charge.Hirshfeld"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["Charge.Hirshfeld"] = "true"

        # Charge.Voronoi - always write with default marker
        if not self.charge_voronoi:
            fdf["Charge.Voronoi"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["Charge.Voronoi"] = "true"

        # COOP.Write - always write with default marker
        if not self.coop_write:
            fdf["COOP.Write"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["COOP.Write"] = "true"

        return fdf

    @classmethod
    def setup_chemical_analysis(
        cls,
        user_params: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG003 kept for interface compatibility
    ) -> "ChemicalAnalysis":
        """
        Create and configure a ChemicalAnalysis instance with full parameter parsing.

        Args:
            user_params: Dictionary of user-defined parameters (case-insensitive,
                may include dots).
            **kwargs: Additional keyword arguments to override or supplement
                user_params.

        Returns
        -------
            ChemicalAnalysis: Configured instance with all fields set.
        """
        # Initialize instance with defaults
        instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            return instance

        # Call update_from_fdf to handle parameter parsing
        instance.update_from_fdf(user_params)

        # Generate FDF
        instance.generate_fdf()

        return instance

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have chemical analysis parameters
        # These are SIESTA-specific post-processing options
        return {}

    def generate_chemical_analysis_block(self) -> None:
        """Generate the chemical analysis options block for the FDF file."""
        logger.info("ChemicalAnalysis.generate_chemical_analysis_block()")
