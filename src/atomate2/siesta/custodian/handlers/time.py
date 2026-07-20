"""Time limit error handler."""

from __future__ import annotations

import logging
from pathlib import Path

from atomate2.siesta.custodian.errors import ErrorType, detect_error
from atomate2.siesta.custodian.fdf_utils import update_fdf_file
from atomate2.siesta.custodian.handlers.base import ErrorHandler

logger = logging.getLogger(__name__)


class TimeHandler(ErrorHandler):
    """Handle wall time limit errors.

    Attempts to restart from saved density matrix and structure.

    This handler inherits from custodian.custodian.ErrorHandler
    and uses custodian's automatic correction tracking.
    """

    # Class properties for custodian library
    is_monitor = False
    is_terminating = True
    raises_runtime_error = True
    max_num_corrections = 2
    raise_on_max = False

    def __init__(self, max_attempts: int = 2) -> None:
        """Initialize TimeHandler.

        Parameters
        ----------
        max_attempts : int, optional
            Maximum correction attempts (default: 2)
        """
        # Store for MSONable serialization
        self.max_attempts = max_attempts
        # Set max_num_corrections for custodian
        self.max_num_corrections = max_attempts
        self.error_type = ErrorType.TIME_LIMIT

    def check(self, directory: str = "./") -> bool:
        """Check for time limit errors.

        Parameters
        ----------
        directory : str, optional
            Directory containing calculation outputs (default: "./")

        Returns
        -------
        bool
            True if time limit error detected, False otherwise
        """
        directory = Path(directory)
        errors = detect_error(directory)
        return any(error.error_type == ErrorType.TIME_LIMIT for error in errors)

    def correct(self, directory: str = "./") -> dict:
        """Apply time limit corrections.

        Parameters
        ----------
        directory : str, optional
            Directory containing calculation outputs (default: "./")

        Returns
        -------
        dict
            Custodian format: {"errors": [...], "actions": [...]}
        """
        directory = Path(directory)
        corrections = {}
        actions = []

        attempt = self.n_applied_corrections + 1

        logger.info(
            f"Applying time limit correction (attempt {attempt}/"
            f"{self.max_num_corrections})"
        )

        # Check for saved files to restart from
        dm_file = directory / "siesta.DM"
        xv_file = directory / "siesta.XV"

        if dm_file.exists():
            corrections["DM.UseSaveDM"] = True
            corrections["DM.Reuse"] = True
            actions.append("Restart from saved density matrix")
            logger.info("  Strategy: Restart from saved density matrix")

        if xv_file.exists():
            corrections["MD.UseSaveXV"] = True
            actions.append("Restart from saved structure")
            logger.info("  Strategy: Restart from saved structure")

        if not corrections:
            logger.warning("  No restart files found, cannot recover")
            return {
                "errors": ["Time limit exceeded"],
                "actions": None,  # Unfixable
            }

        # Apply corrections to FDF file
        fdf_file = directory / "siesta.fdf"
        update_fdf_file(fdf_file, corrections)

        return {
            "errors": ["Time limit exceeded"],
            "actions": actions,
        }
