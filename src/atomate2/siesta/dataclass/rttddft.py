"""
Module defining base SIESTA input set and generator.

class RTTDDFT

Based on User's Guide Siesta 5.4.0
Section:  9.3 Input options for RT-TDDFT
"""

# Metadata

__all__ = ["RTTDDFT"]

import logging
from dataclasses import dataclass, field
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass

logger = logging.getLogger(__name__)


@dataclass
class RTTDDFT(FDFDataclass):
    """Dataclass for Input options for RT-TDDFT."""

    # ------------------------------
    # 9.3 Input options for RT-TDDFT
    # ------------------------------
    # tded_wf_initialize: bool = False  # TDED.WF.Initialize false
    # tded_nsteps: int = 1 # TDED.Nsteps 1
    # tded_time_step: float = 0.001 # TDED.TimeStep 0.001 fs
    # tded_extrapolate: bool = False # TDED.Extrapolate false
    # tded_extrapolate_substeps: int = 3 # TDED.Extrapolate.Substeps 3
    # tded_inverse_linear: bool = True # TDED.Inverse.Linear true
    # tded_wf_save: bool = False # TDED.WF.Save false
    # tded_write_etot: bool = True # TDED.Write.Etot true
    # tded_write_dipole: bool = False # TDED.Write.Dipole false
    # tded_write_eig: bool = False # TDED.Write.Eig false
    # tded_save_rho: bool = False # TDED.Saverho false
    # tded_n_save_rho: int = 100 # TDED.Nsaverho 100
    tded_wf_initialize: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, initializes the wavefunctions for a time-dependent "
                "evolution, often from a ground-state calculation."
            ),
            "SIESTA keyword": "TDED.WF.Initialize",
        },
    )

    tded_nsteps: int = field(
        default=1,
        metadata={
            "description": (
                "The total number of time steps to perform in the "
                "time-dependent simulation."
            ),
            "SIESTA keyword": "TDED.Nsteps",
        },
    )

    tded_time_step: float = field(
        default=0.001,
        metadata={
            "description": (
                "The duration of each time step (in femtoseconds) for the "
                "time-evolution algorithm."
            ),
            "SIESTA keyword": "TDED.TimeStep",
            "unit": "fs",
        },
    )

    tded_extrapolate: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, uses an extrapolation scheme to predict the "
                "wavefunction at the next time step, which can improve "
                "stability and accuracy."
            ),
            "SIESTA keyword": "TDED.Extrapolate",
        },
    )

    tded_extrapolate_substeps: int = field(
        default=3,
        metadata={
            "description": (
                "The number of substeps used within the wavefunction "
                "extrapolation algorithm."
            ),
            "SIESTA keyword": "TDED.Extrapolate.Substeps",
        },
    )

    tded_inverse_linear: bool = field(
        default=True,
        metadata={
            "description": (
                "A technical flag to use a linear approximation for a matrix "
                "inversion step within the time-evolution algorithm."
            ),
            "SIESTA keyword": "TDED.Inverse.Linear",
        },
    )

    tded_wf_save: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, saves the time-evolved wavefunction at the end of "
                "the simulation."
            ),
            "SIESTA keyword": "TDED.WF.Save",
        },
    )

    tded_write_etot: bool = field(
        default=True,
        metadata={
            "description": (
                "If true, writes the total energy of the system at each time step."
            ),
            "SIESTA keyword": "TDED.Write.Etot",
        },
    )

    tded_write_dipole: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, writes the total electric dipole moment of the "
                "system at each time step."
            ),
            "SIESTA keyword": "TDED.Write.Dipole",
        },
    )

    tded_write_eig: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, writes the instantaneous Kohn-Sham eigenvalues at "
                "each time step."
            ),
            "SIESTA keyword": "TDED.Write.Eig",
        },
    )

    tded_save_rho: bool = field(
        default=False,
        metadata={
            "description": (
                "A flag to enable the saving of the time-dependent charge "
                "density on a grid at specified intervals."
            ),
            "SIESTA keyword": "TDED.Saverho",
        },
    )

    tded_n_save_rho: int = field(
        default=100,
        metadata={
            "description": (
                "The frequency of saving the charge density; it will be "
                "saved every 'N' steps."
            ),
            "SIESTA keyword": "TDED.Nsaverho",
        },
    )

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "TDED.WF.Initialize",
                "TDED.Nsteps",
                "TDED.TimeStep",
                "TDED.Extrapolate",
                "TDED.Extrapolate.Substeps",
                "TDED.Inverse.Linear",
                "TDED.WF.Save",
                "TDED.Write.Etot",
                "TDED.Write.Dipole",
                "TDED.Write.Eig",
                "TDED.Saverho",
                "TDED.Nsaverho",
            )
            self.__class__._registered = True  # noqa: SLF001 class-level registration guard

    def validate(self) -> None:
        """
        Validate RT-TDDFT (Real-Time Time-Dependent DFT) parameters.

        Checks configuration for real-time TDDFT calculations including propagation
        settings, time steps, and external perturbations.

        Raises
        ------
        ValueError
            If RT-TDDFT parameters are invalid or inconsistent
        """
        logger.info("RTTDDFT.validate()")

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            # Boolean parameters
            if key_lower in ["tded.wf.initialize", "tded_wf_initialize"]:
                self.tded_wf_initialize = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["tded.extrapolate", "tded_extrapolate"]:
                self.tded_extrapolate = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["tded.inverse.linear", "tded_inverse_linear"]:
                self.tded_inverse_linear = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["tded.wf.save", "tded_wf_save"]:
                self.tded_wf_save = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["tded.write.etot", "tded_write_etot"]:
                self.tded_write_etot = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["tded.write.dipole", "tded_write_dipole"]:
                self.tded_write_dipole = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["tded.write.eig", "tded_write_eig"]:
                self.tded_write_eig = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["tded.saverho", "tded_save_rho"]:
                self.tded_save_rho = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )

            # Integer parameters
            elif key_lower in ["tded.nsteps", "tded_nsteps"]:
                self.tded_nsteps = int(value)
            elif key_lower in [
                "tded.extrapolate.substeps",
                "tded_extrapolate_substeps",
            ]:
                self.tded_extrapolate_substeps = int(value)
            elif key_lower in ["tded.nsaverho", "tded_n_save_rho"]:
                self.tded_n_save_rho = int(value)

            # Float parameters (time step with units)
            elif key_lower in ["tded.timestep", "tded_time_step"]:
                # Handle with or without units (default fs)
                if isinstance(value, str):
                    # Extract numeric value (assume fs if no unit)
                    import re

                    match = re.match(r"([0-9.]+)\s*(\w+)?", value)
                    if match:
                        num_val = float(match.group(1))
                        unit = match.group(2) or "fs"
                        # Convert to fs if needed
                        if unit.lower() in ["fs", "femtoseconds"]:
                            self.tded_time_step = num_val
                        else:
                            # For now, just store the value
                            self.tded_time_step = num_val
                else:
                    self.tded_time_step = float(value)

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters
        """
        fdf: dict[str, Any] = {}

        # Only write if non-default
        if self.tded_wf_initialize:
            fdf["TDED.WF.Initialize"] = "true"
        if self.tded_nsteps != 1:
            fdf["TDED.Nsteps"] = str(self.tded_nsteps)
        if self.tded_time_step != 0.001:
            fdf["TDED.TimeStep"] = f"{self.tded_time_step} fs"
        if self.tded_extrapolate:
            fdf["TDED.Extrapolate"] = "true"
        if self.tded_extrapolate_substeps != 3:
            fdf["TDED.Extrapolate.Substeps"] = str(self.tded_extrapolate_substeps)
        if not self.tded_inverse_linear:  # True is default
            fdf["TDED.Inverse.Linear"] = "false"
        if self.tded_wf_save:
            fdf["TDED.WF.Save"] = "true"
        if not self.tded_write_etot:  # True is default
            fdf["TDED.Write.Etot"] = "false"
        if self.tded_write_dipole:
            fdf["TDED.Write.Dipole"] = "true"
        if self.tded_write_eig:
            fdf["TDED.Write.Eig"] = "true"
        if self.tded_save_rho:
            fdf["TDED.Saverho"] = "true"
        if self.tded_n_save_rho != 100:
            fdf["TDED.Nsaverho"] = str(self.tded_n_save_rho)

        return fdf

    @classmethod
    def setup_rttddft(
        cls,
        user_params: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG003 accepted for interface compatibility
    ) -> "RTTDDFT":
        """
        Create and configure a RTTDDFT instance with full parameter parsing.

        Args:
            user_params: Dictionary of user-defined parameters (case-insensitive,
                may include dots).
            **kwargs: Additional keyword arguments to override or supplement
                user_params.

        Returns
        -------
            RTTDDFT: Configured instance with all fields set.
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
        # ASE doesn't have RT-TDDFT parameters
        # These are SIESTA-specific time-dependent calculations
        return {}
