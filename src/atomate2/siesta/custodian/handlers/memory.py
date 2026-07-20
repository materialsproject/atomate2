"""Memory error handler."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from atomate2.siesta.custodian.errors import ErrorType, detect_error
from atomate2.siesta.custodian.fdf_utils import update_fdf_file
from atomate2.siesta.custodian.handlers.base import ErrorHandler

logger = logging.getLogger(__name__)


class MemoryHandler(ErrorHandler):
    """Handle out-of-memory errors.

    Applies corrections to reduce memory usage:
    1. Reduce diagonalization memory
    2. Enable parallel over k-points
    3. Reduce mesh cutoff
    4. Reduce basis size

    This handler inherits from custodian.custodian.ErrorHandler
    and uses custodian's automatic correction tracking.
    """

    # Class properties for custodian library
    is_monitor = False
    is_terminating = True
    raises_runtime_error = True
    max_num_corrections = 4
    raise_on_max = False

    def __init__(self, max_attempts: int = 4) -> None:
        """Initialize MemoryHandler.

        Parameters
        ----------
        max_attempts : int, optional
            Maximum correction attempts (default: 4)
        """
        # Store for MSONable serialization
        self.max_attempts = max_attempts
        # Set max_num_corrections for custodian
        self.max_num_corrections = max_attempts
        self.error_type = ErrorType.MEMORY

    def check(self, directory: str = "./") -> bool:
        """Check for memory errors.

        Parameters
        ----------
        directory : str, optional
            Directory containing SIESTA output (default: "./")

        Returns
        -------
        bool
            True if memory error detected, False otherwise
        """
        directory = Path(directory)
        errors = detect_error(directory)
        return any(error.error_type == ErrorType.MEMORY for error in errors)

    def correct(self, directory: str = "./") -> dict:
        """Apply memory reduction corrections.

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
            f"Applying memory reduction correction (attempt {attempt}/"
            f"{self.max_num_corrections})"
        )

        if attempt == 1:
            # Level 1: Reduce diagonalization memory
            corrections["Diag.Memory"] = 0.5
            strategy = "Reduce Diag.Memory to 50%"

        elif attempt == 2:
            # Level 2: Enable k-point parallelization
            corrections["Diag.Memory"] = 0.3
            corrections["Diag.ParallelOverK"] = True
            strategy = "Enable ParallelOverK, reduce memory to 30%"

        elif attempt == 3:
            # Level 3: Reduce mesh cutoff
            current_cutoff = self._get_current_cutoff(directory)
            new_cutoff = max(200.0, current_cutoff * 0.85)  # Reduce by 15%
            corrections["Diag.Memory"] = 0.3
            corrections["Diag.ParallelOverK"] = True
            corrections["MeshCutoff"] = f"{new_cutoff:.1f} Ry"  # type: ignore[assignment]
            strategy = f"Reduce mesh cutoff to {new_cutoff:.1f} Ry (15% reduction)"

        else:
            # Level 4: Reduce basis size (last resort)
            current_basis = self._get_current_basis(directory)
            new_basis = self._reduce_basis_size(current_basis)
            corrections["Diag.Memory"] = 0.3
            corrections["Diag.ParallelOverK"] = True
            corrections["PAO.BasisSize"] = new_basis  # type: ignore[assignment]
            strategy = f"Reduce basis from {current_basis} to {new_basis}"

        logger.info(f"  Strategy: {strategy}")

        # Apply corrections to FDF file
        fdf_file = directory / "siesta.fdf"
        update_fdf_file(fdf_file, corrections)

        return {
            "errors": ["Out of memory error"],
            "actions": [f"Level {attempt}: {strategy}"],
        }

    def _get_current_cutoff(self, directory: Path) -> float:
        """Extract current mesh cutoff from siesta.fdf.

        Parameters
        ----------
        directory : Path
            Directory containing siesta.fdf

        Returns
        -------
        float
            Mesh cutoff in Ry (default 300.0 if not found)
        """
        fdf_file = directory / "siesta.fdf"
        if not fdf_file.exists():
            return 300.0

        with open(fdf_file) as f:
            for line in f:
                match = re.search(r"MeshCutoff\s+([0-9.]+)\s*Ry", line, re.IGNORECASE)
                if match:
                    return float(match.group(1))

        return 300.0  # Default

    def _get_current_basis(self, directory: Path) -> str:
        """Extract current basis size from siesta.fdf.

        Parameters
        ----------
        directory : Path
            Directory containing siesta.fdf

        Returns
        -------
        str
            Basis size (default "DZP" if not found)
        """
        fdf_file = directory / "siesta.fdf"
        if not fdf_file.exists():
            return "DZP"

        with open(fdf_file) as f:
            for line in f:
                match = re.search(r"PAO\.BasisSize\s+(\w+)", line, re.IGNORECASE)
                if match:
                    return match.group(1).upper()

        return "DZP"  # Default

    def _reduce_basis_size(self, current: str) -> str:
        """Get reduced basis size.

        Parameters
        ----------
        current : str
            Current basis size

        Returns
        -------
        str
            Smaller basis size
        """
        hierarchy = ["SZ", "DZ", "SZP", "DZP", "DZDP", "TZP", "TZDP"]
        try:
            idx = hierarchy.index(current.upper())
            if idx > 0:
                return hierarchy[idx - 1]
        except (ValueError, IndexError):
            pass
        return "DZ"  # Safe fallback
