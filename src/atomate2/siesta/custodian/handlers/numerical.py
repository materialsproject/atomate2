"""Numerical instability error handler."""

from __future__ import annotations

import logging
from pathlib import Path

from atomate2.siesta.custodian.errors import ErrorType, detect_error
from atomate2.siesta.custodian.fdf_utils import update_fdf_file
from atomate2.siesta.custodian.handlers.base import ErrorHandler

logger = logging.getLogger(__name__)


class NumericalHandler(ErrorHandler):
    """Handle numerical instability errors.

    Applies corrections to improve numerical stability:
    1. Tighten SCF tolerance
    2. Increase mesh cutoff
    3. Use more confined basis

    This handler inherits from custodian.custodian.ErrorHandler
    and uses custodian's automatic correction tracking.
    """

    # Class properties for custodian library
    is_monitor = False
    is_terminating = True
    raises_runtime_error = True
    max_num_corrections = 3
    raise_on_max = False

    def __init__(self, max_attempts: int = 3) -> None:
        """Initialize NumericalHandler.

        Parameters
        ----------
        max_attempts : int, optional
            Maximum correction attempts (default: 3)
        """
        # Store for MSONable serialization
        self.max_attempts = max_attempts
        # Set max_num_corrections for custodian
        self.max_num_corrections = max_attempts
        self.error_type = ErrorType.NUMERICAL

    def check(self, directory: str = "./") -> bool:
        """Check for numerical errors.

        Parameters
        ----------
        directory : str, optional
            Directory containing SIESTA output (default: "./")

        Returns
        -------
        bool
            True if numerical error detected, False otherwise
        """
        directory = Path(directory)
        errors = detect_error(directory)
        return any(error.error_type == ErrorType.NUMERICAL for error in errors)

    def correct(self, directory: str = "./") -> dict:
        """Apply numerical stability corrections.

        Parameters
        ----------
        directory : str, optional
            Directory containing SIESTA output (default: "./")

        Returns
        -------
        dict
            Custodian format: {"errors": [...], "actions": [...]}
        """
        directory = Path(directory)
        corrections = {}

        attempt = self.n_applied_corrections + 1

        logger.info(
            f"Applying numerical stability correction (attempt {attempt}/"
            f"{self.max_num_corrections})"
        )

        if attempt == 1:
            # Level 1: Tighten SCF tolerance
            corrections["DM.Tolerance"] = 1.0e-6
            corrections["SCF.Mixer.Weight"] = 0.01
            strategy = "Tighten DM tolerance to 1e-6"

        elif attempt == 2:
            # Level 2: Increase mesh cutoff for better accuracy
            corrections["DM.Tolerance"] = 1.0e-6
            corrections["MeshCutoff"] = "400 Ry"  # type: ignore[assignment]
            strategy = "Increase mesh cutoff to 400 Ry"

        else:
            # Level 3: Use more confined basis
            corrections["DM.Tolerance"] = 1.0e-6
            corrections["MeshCutoff"] = "400 Ry"  # type: ignore[assignment]
            corrections["PAO.EnergyShift"] = 0.02
            strategy = "Increase EnergyShift for better confinement"

        logger.info(f"  Strategy: {strategy}")

        # Apply corrections to FDF file
        fdf_file = directory / "siesta.fdf"
        update_fdf_file(fdf_file, corrections)

        return {
            "errors": ["Numerical instability detected (NaN/Inf values)"],
            "actions": [f"Level {attempt}: {strategy}"],
        }
