"""Basis set error handler for SIESTA calculations."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from atomate2.siesta.custodian.errors import ErrorType, detect_error
from atomate2.siesta.custodian.fdf_utils import update_fdf_file
from atomate2.siesta.custodian.handlers.base import ErrorHandler

logger = logging.getLogger(__name__)


class BasisSetHandler(ErrorHandler):
    """Handle basis set generation errors.

    Applies corrections for common basis set issues:
    1. Split-norm degenerate error (hydrogen with DZP/TZP)
    2. Orbital confinement issues
    3. Basis extension problems

    For hydrogen split-norm errors, reduces basis to SZ or SZP.
    For other basis errors, adjusts PAO.EnergyShift and SplitNorm.

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
        """Initialize BasisSetHandler.

        Parameters
        ----------
        max_attempts : int, optional
            Maximum correction attempts (default: 3)
        """
        # Store for MSONable serialization
        self.max_attempts = max_attempts
        # Set max_num_corrections for custodian
        self.max_num_corrections = max_attempts
        self.error_type = ErrorType.BASIS

    def check(self, directory: str = "./") -> bool:
        """Check for basis set errors.

        Parameters
        ----------
        directory : str, optional
            Directory containing SIESTA output (default: "./")

        Returns
        -------
        bool
            True if basis set error detected, False otherwise
        """
        dir_path = Path(directory)
        errors = detect_error(dir_path)
        return any(error.error_type == ErrorType.BASIS for error in errors)

    def correct(self, directory: str = "./") -> dict:
        """Apply basis set corrections.

        Parameters
        ----------
        directory : str, optional
            Directory containing SIESTA output (default: "./")

        Returns
        -------
        dict
            Custodian format: {"errors": [...], "actions": [...]}
        """
        dir_path = Path(directory)
        corrections = {}

        attempt = self.n_applied_corrections + 1

        logger.info(
            f"Applying basis set correction (attempt {attempt}/"
            f"{self.max_num_corrections})"
        )

        # Check if this is a hydrogen split-norm error
        output_file = dir_path / "siesta.out"
        is_hydrogen_error = False
        if output_file.exists():
            with open(output_file) as f:
                content = f.read()
                if re.search(r"Split-norm parameter is too small.*degenerate", content):
                    is_hydrogen_error = True

        if is_hydrogen_error:
            # Hydrogen-specific correction: reduce basis size
            logger.info("  Detected hydrogen split-norm error")
            corrections = self._fix_hydrogen_basis(dir_path, attempt)
            strategy = corrections.get("strategy", "Reduce hydrogen basis")
        else:
            # General basis corrections
            corrections = self._fix_general_basis(attempt)
            strategy = corrections.get("strategy", "Adjust basis parameters")

        logger.info(f"  Strategy: {strategy}")

        # Apply corrections to FDF file
        fdf_file = dir_path / "siesta.fdf"
        update_params = {k: v for k, v in corrections.items() if k != "strategy"}
        if update_params:
            update_fdf_file(fdf_file, update_params)

        return {
            "errors": ["Basis set generation error"],
            "actions": [f"Level {attempt}: {strategy}"],
        }

    def _fix_hydrogen_basis(self, directory: Path, attempt: int) -> dict:
        """Fix hydrogen split-norm errors by reducing basis size.

        Parameters
        ----------
        directory : Path
            Job directory
        attempt : int
            Correction attempt number

        Returns
        -------
        dict
            Corrections dictionary with PAO.Basis block
        """
        corrections: dict = {}

        if attempt == 1:
            # Try DZP for hydrogen only
            # H     1     # Species label, number of l-shells
            # n=1   0   2 P 1   # n, l, Nzeta, Polarization, NzetaPol
            #  5.0
            corrections["PAO.Basis"] = """PAO.Basis
%block PAO.Basis
H                     2                    # Species label, number of l-shells
 n=1   0   2                         # n, l, Nzeta
   5.233      3.529
   1.000      1.000
 n=2   1   1                         # n, l, Nzeta
   5.233
   1.000
%endblock PAO.Basis"""
            corrections["strategy"] = "Use SZP for hydrogen, keep current for others"

        elif attempt == 2:
            # Try SZ for hydrogen (no polarization)
            corrections["PAO.Basis"] = """PAO.Basis
%block PAO.Basis
H     1
 n=1   0   1   # SZ: Single zeta, no polarization
   5.0
%endblock PAO.Basis"""
            corrections["strategy"] = "Use SZ for hydrogen (no polarization)"

        else:
            # Last resort: reduce all to SZ
            corrections["PAO.BasisSize"] = "SZ"
            corrections["PAO.SplitNorm"] = 0.30  # type: ignore[assignment]
            corrections["strategy"] = "Reduce all atoms to SZ with larger split-norm"

        return corrections

    def _fix_general_basis(self, attempt: int) -> dict:
        """Fix general basis set errors.

        Parameters
        ----------
        attempt : int
            Correction attempt number

        Returns
        -------
        dict
            Corrections dictionary
        """
        corrections: dict = {}

        if attempt == 1:
            # Level 1: Increase energy shift for better confinement
            corrections["PAO.EnergyShift"] = 0.02  # type: ignore[assignment]
            corrections["strategy"] = "Increase PAO.EnergyShift to 0.02 eV"

        elif attempt == 2:
            # Level 2: Adjust split-norm
            corrections["PAO.EnergyShift"] = 0.02  # type: ignore[assignment]
            corrections["PAO.SplitNorm"] = 0.30  # type: ignore[assignment]
            corrections["strategy"] = "Adjust split-norm to 0.30"

        else:
            # Level 3: Use more confined basis with explicit soft confinement
            corrections["PAO.EnergyShift"] = 0.05  # type: ignore[assignment]
            corrections["PAO.SplitNorm"] = 0.40  # type: ignore[assignment]
            corrections["PAO.BasisSize"] = "DZP"
            corrections["PAO.SoftDefault"] = "T"
            corrections["PAO.SoftInnerRadius"] = 0.90  # type: ignore[assignment]
            corrections["PAO.SoftPotential"] = 40.0  # type: ignore[assignment]
            corrections["strategy"] = "Use soft confinement with larger parameters"

        return corrections
