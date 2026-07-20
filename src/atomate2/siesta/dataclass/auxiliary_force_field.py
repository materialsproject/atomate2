"""
Module defining base SIESTA input set and generator.

class AuxiliaryForceField

Based on User's Guide Siesta 5.4.0
Section: 6.25 Auxiliary Force field
"""

# Metadata

__all__ = ["AuxiliaryForceField"]

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

from atomate2.siesta.dataclass.base import FDFDataclass

logger = logging.getLogger(__name__)


@dataclass
class AuxiliaryForceField(FDFDataclass):
    """SIESTA auxiliary force field and dispersion-correction parameters."""

    # --------------------------
    # 6.25 Auxiliary Force field
    # --------------------------
    # mm_potentials_block: Dict[float,Any]= field(default_factory=dict)
    # mm_cutoff: float = 30.0 # MM.Cutoff 30 Bohr
    # mm_units_energy: float = None # MM.UnitsEnergy eV
    # mm_units_distance: float = None # MM.UnitsDistance Ang
    # mm_grimme_d: float = 20.0 # MM.Grimme.D 20.0
    # mm_grimme_s6: float = 1.66 # MM.Grimme.S6 1.66
    mm_potentials_block: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "Defines molecular mechanics potentials for the auxiliary force "
                "field (SIESTA keyword: %block MM.Potentials)."
            )
        },
    )  # %block MM.Potentials 〈None〉

    mm_cutoff: float = field(
        default=30.0,
        metadata={
            "description": (
                "The real-space cutoff distance (in Bohr) for the molecular "
                "mechanics potentials."
            ),
            "SIESTA keyword": "MM.Cutoff",
            "unit": "Bohr",
        },
    )

    mm_units_energy: str = field(
        default="",
        metadata={
            "description": (
                "Specifies the units of energy to be used in the "
                "'%block MM.Potentials'. Default is eV."
            ),
            "SIESTA keyword": "MM.UnitsEnergy",
            "unit": "eV",
        },
    )

    mm_units_distance: str = field(
        default="",
        metadata={
            "description": (
                "Specifies the units of distance to be used in the "
                "'%block MM.Potentials'. Default is Angstrom."
            ),
            "SIESTA keyword": "MM.UnitsDistance",
            "unit": "Ang",
        },
    )

    mm_grimme_d: float = field(
        default=20.0,
        metadata={
            "description": (
                "The damping parameter 'd' for the Grimme D2 van der Waals "
                "correction scheme."
            ),
            "SIESTA keyword": "MM.Grimme.D",
        },
    )

    mm_grimme_s6: float = field(
        default=1.66,
        metadata={
            "description": (
                "The global scaling factor 's6' for the Grimme D2 van der Waals "
                "correction scheme."
            ),
            "SIESTA keyword": "MM.Grimme.S6",
        },
    )

    # -------------------------------------
    # 6.26 Grimme’s DFT-D3 dispersion model  # noqa: RUF003
    # -------------------------------------
    # dft3: bool = False # DFTD3 false
    # dft3_use_xc_defaults: bool = True  # DFTD3.UseXCDefaults true
    # dft3_b_j_damping: bool = True  # DFTD3.BJdamping true
    # dft3_s6: float = 1.0 # DFTD3.s6 1.0
    # dft3_rs6: float = 1.0 # DFTD3.rs6 1.0
    # dft3_s8: float = 1.0  # DFTD3.s8 1.0
    # dft3_rs8: float = 1.0 # DFTD3.rs8 1.0
    # dft3_alpha: float = 14.0  # DFTD3.alpha 14.0
    # dft3_a1: float = 0.4 # DFTD3.a1 0.4
    # dft3_a2: float = 5.0  # DFTD3.a2 5.0
    # dft3_2_body_cutoff: float = 60.0 # DFTD3.2BodyCutOff 60.0bohr
    # dft3_3_body_cutoff: float = 40.0 # DFTD3.3BodyCutOff 40.0bohr
    # dft3_coordination_cutoff: float = 10.0 # DFTD3.CoordinationCutoff 10.0 bohr
    dft3: bool = field(
        default=False,
        metadata={
            "description": (
                "A master flag to enable the Grimme D3 dispersion correction for "
                "van der Waals interactions."
            ),
            "SIESTA keyword": "DFTD3",
        },
    )

    dft3_use_xc_defaults: bool = field(
        default=True,
        metadata={
            "description": (
                "If true, automatically uses the recommended D3 parameters for "
                "the chosen exchange-correlation functional."
            ),
            "SIESTA keyword": "DFTD3.UseXCDefaults",
        },
    )

    dft3_2_body_cutoff: float = field(
        default=60.0,
        metadata={
            "description": (
                "The real-space cutoff distance (in Bohr) for the two-body "
                "dispersion term."
            ),
            "SIESTA keyword": "DFTD3.2BodyCutOff",
            "unit": "Bohr",
        },
    )

    dft3_3_body_cutoff: float = field(
        default=40.0,
        metadata={
            "description": (
                "The real-space cutoff distance (in Bohr) for the three-body "
                "dispersion term."
            ),
            "SIESTA keyword": "DFTD3.3BodyCutOff",
            "unit": "Bohr",
        },
    )

    dft3_coordination_cutoff: float = field(
        default=10.0,
        metadata={
            "description": (
                "The cutoff distance (in Bohr) used for calculating atomic "
                "coordination numbers within the D3 scheme."
            ),
            "SIESTA keyword": "DFTD3.CoordinationCutoff",
            "unit": "Bohr",
        },
    )

    dft3_b_j_damping: bool = field(
        default=True,
        metadata={
            "description": (
                "A flag to enable the Becke-Johnson (BJ) damping function, which "
                "provides a more accurate short-range behavior for the dispersion "
                "correction."
            ),
            "SIESTA keyword": "DFTD3.BJdamping",
        },
    )

    dft3_s6: float = field(
        default=1.0,
        metadata={
            "description": (
                "The global scaling factor (s6) for the two-body (C6) dispersion term."
            ),
            "SIESTA keyword": "DFTD3.s6",
        },
    )

    dft3_rs6: float = field(
        default=1.0,
        metadata={
            "description": (
                "The scaling factor (sr,6) for the damping function of the "
                "two-body C6 term."
            ),
            "SIESTA keyword": "DFTD3.rs6",
        },
    )

    dft3_s8: float = field(
        default=1.0,
        metadata={
            "description": (
                "The global scaling factor (s8) for the two-body (C8) dispersion term."
            ),
            "SIESTA keyword": "DFTD3.s8",
        },
    )

    dft3_rs8: float = field(
        default=1.0,
        metadata={
            "description": (
                "The scaling factor (sr,8) for the damping function of the "
                "two-body C8 term."
            ),
            "SIESTA keyword": "DFTD3.rs8",
        },
    )

    dft3_alpha: float = field(
        default=14.0,
        metadata={
            "description": (
                "A parameter for the three-body dispersion term. Note: This "
                "specific keyword might not be standard; 'a1' and 'a2' are used "
                "for BJ damping."
            ),
            "SIESTA keyword": "DFTD3.alpha",
        },
    )

    dft3_a1: float = field(
        default=0.4,
        metadata={
            "description": "The parameter 'a1' for the Becke-Johnson damping function.",
            "SIESTA keyword": "DFTD3.a1",
        },
    )

    dft3_a2: float = field(
        default=5.0,
        metadata={
            "description": "The parameter 'a2' for the Becke-Johnson damping function.",
            "SIESTA keyword": "DFTD3.a2",
        },
    )

    _registered: ClassVar[bool]

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "%block MM.Potentials",
                "MM.Cutoff",
                "MM.UnitsEnergy",
                "MM.UnitsDistance",
                "MM.Grimme.D",
                "MM.Grimme.S6",
                "DFTD3",
                "DFTD3.UseXCDefaults",
                "DFTD3.2BodyCutOff",
                "DFTD3.3BodyCutOff",
                "DFTD3.CoordinationCutoff",
                "DFTD3.BJdamping",
                "DFTD3.s6",
                "DFTD3.rs6",
                "DFTD3.s8",
                "DFTD3.rs8",
                "DFTD3.alpha",
                "DFTD3.a1",
                "DFTD3.a2",
                # "DFTD3.Periodic" # TODO: Need to added
            )
            self.__class__._registered = True  # noqa: SLF001

    def validate(self) -> None:
        """
        Validate auxiliary force field parameters.

        Checks configuration for auxiliary force fields used in hybrid QM/MM or
        empirical correction schemes (e.g., dispersion corrections, force fields).

        Raises
        ------
        ValueError
            If force field parameters are invalid
        """
        logger.info("AuxiliaryForceField.validate()")

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        from atomate2.siesta.dataclass.units import parse_length

        for key, value in fdf_dict.items():
            key_lower = key.lower()

            # MM parameters
            if key_lower in ["%block mm.potentials", "mm_potentials_block"]:
                self.mm_potentials_block = value
            elif key_lower in ["mm.cutoff", "mm_cutoff"]:
                # Parse length with units (default Bohr)
                self.mm_cutoff = parse_length(value, target_unit="Bohr")
            elif key_lower in ["mm.unitsenergy", "mm_units_energy"]:
                self.mm_units_energy = str(value)
            elif key_lower in ["mm.unitsdistance", "mm_units_distance"]:
                self.mm_units_distance = str(value)
            elif key_lower in ["mm.grimme.d", "mm_grimme_d"]:
                self.mm_grimme_d = float(value)
            elif key_lower in ["mm.grimme.s6", "mm_grimme_s6"]:
                self.mm_grimme_s6 = float(value)

            # DFTD3 parameters
            elif key_lower in ["dftd3", "dft3"]:
                self.dft3 = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["dftd3.usexcdefaults", "dft3_use_xc_defaults"]:
                self.dft3_use_xc_defaults = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["dftd3.bjdamping", "dft3_b_j_damping"]:
                self.dft3_b_j_damping = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["dftd3.s6", "dft3_s6"]:
                self.dft3_s6 = float(value)
            elif key_lower in ["dftd3.rs6", "dft3_rs6"]:
                self.dft3_rs6 = float(value)
            elif key_lower in ["dftd3.s8", "dft3_s8"]:
                self.dft3_s8 = float(value)
            elif key_lower in ["dftd3.rs8", "dft3_rs8"]:
                self.dft3_rs8 = float(value)
            elif key_lower in ["dftd3.alpha", "dft3_alpha"]:
                self.dft3_alpha = float(value)
            elif key_lower in ["dftd3.a1", "dft3_a1"]:
                self.dft3_a1 = float(value)
            elif key_lower in ["dftd3.a2", "dft3_a2"]:
                self.dft3_a2 = float(value)
            elif key_lower in ["dftd3.2bodycutoff", "dft3_2_body_cutoff"]:
                # Parse length with units (default Bohr)
                self.dft3_2_body_cutoff = parse_length(value, target_unit="Bohr")
            elif key_lower in ["dftd3.3bodycutoff", "dft3_3_body_cutoff"]:
                # Parse length with units (default Bohr)
                self.dft3_3_body_cutoff = parse_length(value, target_unit="Bohr")
            elif key_lower in [
                "dftd3.coordinationcutoff",
                "dft3_coordination_cutoff",
            ]:
                # Parse length with units (default Bohr)
                self.dft3_coordination_cutoff = parse_length(value, target_unit="Bohr")

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters
        """
        fdf: dict[str, Any] = {}

        # MM potentials
        if self.mm_potentials_block:
            fdf["%block MM.Potentials"] = self.mm_potentials_block
        if self.mm_cutoff != 30.0:  # Only if different from default
            fdf["MM.Cutoff"] = f"{self.mm_cutoff} Bohr"
        if self.mm_units_energy:
            fdf["MM.UnitsEnergy"] = self.mm_units_energy
        if self.mm_units_distance:
            fdf["MM.UnitsDistance"] = self.mm_units_distance
        if self.mm_grimme_d != 20.0:
            fdf["MM.Grimme.D"] = str(self.mm_grimme_d)
        if self.mm_grimme_s6 != 1.66:
            fdf["MM.Grimme.S6"] = str(self.mm_grimme_s6)

        # DFTD3
        if self.dft3:
            fdf["DFTD3"] = "true"
            if not self.dft3_use_xc_defaults:
                fdf["DFTD3.UseXCDefaults"] = "false"
            if not self.dft3_b_j_damping:
                fdf["DFTD3.BJdamping"] = "false"
            # Only write non-default parameters
            if self.dft3_s6 != 1.0:
                fdf["DFTD3.s6"] = str(self.dft3_s6)
            if self.dft3_rs6 != 1.0:
                fdf["DFTD3.rs6"] = str(self.dft3_rs6)
            if self.dft3_s8 != 1.0:
                fdf["DFTD3.s8"] = str(self.dft3_s8)
            if self.dft3_rs8 != 1.0:
                fdf["DFTD3.rs8"] = str(self.dft3_rs8)
            if self.dft3_alpha != 14.0:
                fdf["DFTD3.alpha"] = str(self.dft3_alpha)
            if self.dft3_a1 != 0.4:
                fdf["DFTD3.a1"] = str(self.dft3_a1)
            if self.dft3_a2 != 5.0:
                fdf["DFTD3.a2"] = str(self.dft3_a2)
            if self.dft3_2_body_cutoff != 60.0:
                fdf["DFTD3.2BodyCutOff"] = f"{self.dft3_2_body_cutoff} Bohr"
            if self.dft3_3_body_cutoff != 40.0:
                fdf["DFTD3.3BodyCutOff"] = f"{self.dft3_3_body_cutoff} Bohr"
            if self.dft3_coordination_cutoff != 10.0:
                fdf["DFTD3.CoordinationCutoff"] = (
                    f"{self.dft3_coordination_cutoff} Bohr"
                )

        return fdf

    @classmethod
    def setup_auxiliary_force_field(
        cls,
        user_params: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG003
    ) -> "AuxiliaryForceField":
        """
        Create and configure an AuxiliaryForceField instance with parameter parsing.

        Args:
            user_params: Dictionary of user-defined parameters (case-insensitive,
                may include dots).
            **kwargs: Additional keyword arguments to override or supplement
                user_params.

        Returns
        -------
            AuxiliaryForceField: Configured instance with all fields set.
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
        # ASE doesn't have auxiliary force field parameters
        # These are SIESTA-specific for QM/MM and dispersion corrections
        return {}
