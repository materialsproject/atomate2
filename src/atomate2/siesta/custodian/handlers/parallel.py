"""Parallel distribution error handler."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from atomate2.siesta.custodian.errors import ErrorType, detect_error
from atomate2.siesta.custodian.fdf_utils import update_fdf_file
from atomate2.siesta.custodian.handlers.base import ErrorHandler

logger = logging.getLogger(__name__)


class ParallelDistributionHandler(ErrorHandler):
    """Handle parallel distribution errors (too many processors for system size).

    When SIESTA reports "You have too many processors for the system size",
    the number of orbitals is insufficient for the MPI process count.

    Recovery strategy:
    1. Increase basis size (SZ → SZP → DZ → DZP → TZP) to add more orbitals
    2. If basis is already at maximum, report that fewer MPI processes are needed

    Since MPI process count cannot be changed mid-run, this handler increases
    the basis size which adds more orbitals for better load distribution.

    Example error messages:
    - "You have too many processors for the system size"
    - "Some processors are idle. Check PARALLEL_DIST"
    - "Orbital distribution balance (max,min): 1 0"

    Notes
    -----
    This is common for small systems (e.g., H₂ with SZ basis = 2 orbitals)
    when running with many MPI processes. The solution is either:
    - Reduce MPI processes (requires job resubmission)
    - Increase basis size (this handler's approach)
    - Use larger supercell (not always possible)
    """

    # Class properties for custodian library
    is_monitor = False
    is_terminating = True
    raises_runtime_error = True
    max_num_corrections = 4
    raise_on_max = True  # Raise if we can't fix it

    # Basis size hierarchy (increasing orbital count)
    BASIS_HIERARCHY = ["SZ", "SZP", "DZ", "DZP", "TZP", "TZDP"]

    def __init__(self, max_attempts: int = 4):
        """Initialize ParallelDistributionHandler.

        Parameters
        ----------
        max_attempts : int, optional
            Maximum correction attempts (default: 4)
        """
        self.max_attempts = max_attempts
        self.max_num_corrections = max_attempts
        self.error_type = ErrorType.PARALLEL

    def check(self, directory="./") -> bool:
        """Check for parallel distribution errors.

        Parameters
        ----------
        directory : str, optional
            Directory containing SIESTA output (default: "./")

        Returns
        -------
        bool
            True if parallel distribution error detected, False otherwise
        """
        directory = Path(directory)
        errors = detect_error(directory)
        for error in errors:
            if error.error_type == ErrorType.PARALLEL:
                logger.warning(
                    "Parallel distribution error detected: "
                    "Too many processors for system size"
                )
                return True
        return False

    def correct(self, directory="./") -> dict:
        """Apply parallel distribution corrections.

        Strategy: Increase basis size to add more orbitals.

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
            f"Applying parallel distribution correction (attempt {attempt}/"
            f"{self.max_num_corrections})"
        )

        # Get current basis size
        current_basis = self._get_current_basis(directory)
        new_basis = self._increase_basis_size(current_basis)

        if new_basis == current_basis:
            # Already at maximum basis, cannot increase further
            n_procs = self._estimate_mpi_processes(directory)
            n_orbitals = self._estimate_orbitals(directory)

            error_msg = (
                f"Cannot fix parallel distribution error: "
                f"Basis already at {current_basis} (max), "
                f"but system has only ~{n_orbitals} orbitals for ~{n_procs} MPI processes. "
                f"Please resubmit with fewer MPI processes (recommend: {max(1, n_orbitals // 2)})"
            )
            logger.error(error_msg)

            return {
                "errors": ["Parallel distribution error - cannot recover"],
                "actions": [
                    f"FAILED: {error_msg}",
                    f"Recommendation: Reduce MPI processes to {max(1, n_orbitals // 2)} or less",
                ],
            }

        # Increase basis size
        corrections["PAO.BasisSize"] = new_basis
        strategy = f"Increase basis from {current_basis} to {new_basis} (more orbitals)"

        logger.info(f"  Strategy: {strategy}")
        logger.info(
            "  Rationale: Larger basis = more orbitals = better MPI distribution"
        )

        # Apply corrections to FDF file
        fdf_file = directory / "siesta.fdf"
        update_fdf_file(fdf_file, corrections)

        return {
            "errors": ["Parallel distribution error (too many processors)"],
            "actions": [f"Level {attempt}: {strategy}"],
        }

    def _get_current_basis(self, directory: Path) -> str:
        """Extract current basis size from siesta.fdf.

        Parameters
        ----------
        directory : Path
            Directory containing siesta.fdf

        Returns
        -------
        str
            Basis size (default "SZ" if not found)
        """
        fdf_file = directory / "siesta.fdf"
        if not fdf_file.exists():
            return "SZ"

        with open(fdf_file) as f:
            for line in f:
                match = re.search(r"PAO\.BasisSize\s+(\w+)", line, re.IGNORECASE)
                if match:
                    return match.group(1).upper()

        return "SZ"  # Default for small systems

    def _increase_basis_size(self, current: str) -> str:
        """Get increased basis size.

        Parameters
        ----------
        current : str
            Current basis size

        Returns
        -------
        str
            Larger basis size, or same if already at max
        """
        try:
            idx = self.BASIS_HIERARCHY.index(current.upper())
            if idx < len(self.BASIS_HIERARCHY) - 1:
                return self.BASIS_HIERARCHY[idx + 1]
        except (ValueError, IndexError):
            pass
        return current  # Already at max or unknown

    def _estimate_mpi_processes(self, directory: Path) -> int:
        """Estimate number of MPI processes from SIESTA output.

        Parameters
        ----------
        directory : Path
            Directory containing siesta.out

        Returns
        -------
        int
            Estimated MPI process count (default 4)
        """
        out_file = directory / "siesta.out"
        if not out_file.exists():
            return 4

        try:
            with open(out_file) as f:
                content = f.read()
                # Look for "Stopping Program from Node: X"
                matches = re.findall(r"Stopping Program from Node:\s*(\d+)", content)
                if matches:
                    return max(int(m) for m in matches) + 1

                # Look for "Running on X nodes"
                match = re.search(r"Running on\s+(\d+)\s+nodes", content)
                if match:
                    return int(match.group(1))
        except Exception:
            pass

        return 4  # Default assumption

    def _estimate_orbitals(self, directory: Path) -> int:
        """Estimate number of orbitals from SIESTA output.

        Parameters
        ----------
        directory : Path
            Directory containing siesta.out

        Returns
        -------
        int
            Estimated orbital count (default 2)
        """
        out_file = directory / "siesta.out"
        if not out_file.exists():
            return 2

        try:
            with open(out_file) as f:
                for line in f:
                    # Look for "Total number of orbitals:"
                    match = re.search(
                        r"Total number of.*orbitals:\s*(\d+)", line, re.IGNORECASE
                    )
                    if match:
                        return int(match.group(1))

                    # Look for orbital distribution
                    match = re.search(
                        r"Orbital distribution balance.*:\s*(\d+)\s+(\d+)", line
                    )
                    if match:
                        max_orb = int(match.group(1))
                        return max_orb  # At least this many orbitals
        except Exception:
            pass

        return 2  # Default for tiny systems
