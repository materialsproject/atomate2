"""Geometry convergence error handler for relaxation calculations."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from atomate2.siesta.custodian.fdf_utils import read_fdf_file, update_fdf_file
from atomate2.siesta.custodian.handlers.base import ErrorHandler

logger = logging.getLogger(__name__)


class GeometryConvergenceHandler(ErrorHandler):
    """Handle geometry convergence failures during relaxation calculations.

    This handler detects when a relaxation calculation completes but does
    not reach the requested force tolerance. It applies progressive corrections
    to help difficult systems converge:

    1. Increase MD.NumCGsteps (allow more optimization steps)
    2. Adjust force tolerance temporarily (for intermediate steps)
    3. Try different optimization methods (CG → FIRE → Broyden)

    Why needed?
    -----------
    Some systems require many MD steps to converge:
    - Complex surfaces with many degrees of freedom
    - Systems near saddle points
    - Structures with soft modes
    - Tight force tolerances

    Solution: Progressively increase allowed MD steps and try alternative methods.

    Correction Strategy:
    --------------------
    Level 1: Increase MD.NumCGsteps by 50% (e.g., 200 → 300)
    Level 2: Increase by 100% (e.g., 200 → 400)
    Level 3: Increase by 150% + try FIRE method (faster for some systems)
    Level 4: Increase by 200% + adjust convergence criteria slightly
    Level 5: Maximum steps (1000) + Broyden method (last resort)

    This handler inherits from custodian.custodian.ErrorHandler
    and uses custodian's automatic correction tracking.
    """

    # Class properties for custodian library
    is_monitor = False  # Check errors at end of job
    is_terminating = True  # Terminate job on error detection
    raises_runtime_error = True  # Raise if cannot recover
    max_num_corrections = 5  # Maximum correction attempts
    raise_on_max = False  # Don't raise, just stop correcting

    def __init__(
        self,
        max_attempts: int = 5,
        force_tolerance: float = 0.04,  # eV/Ang
    ):
        """Initialize GeometryConvergenceHandler.

        Parameters
        ----------
        max_attempts : int, optional
            Maximum correction attempts (default: 5)
        force_tolerance : float, optional
            Maximum force tolerance in eV/Ang (default: 0.04)

        Note
        ----
        The max_attempts parameter sets the max_num_corrections
        class property for compatibility with custodian library.
        """
        # Store for MSONable serialization
        self.max_attempts = max_attempts
        self.force_tolerance = force_tolerance
        # Set max_num_corrections for custodian
        self.max_num_corrections = max_attempts

    def check(self, directory="./") -> bool:
        """Check for geometry convergence failures.

        Parameters
        ----------
        directory : str, optional
            Directory containing SIESTA output (default: "./")

        Returns
        -------
        bool
            True if geometry did not converge, False otherwise
        """
        directory = Path(directory)
        output_file = directory / "siesta.out"

        if not output_file.exists():
            return False

        try:
            content = output_file.read_text()
        except Exception as e:
            logger.warning(f"Could not read output file: {e}")
            return False

        # Check if job completed (some geometry steps were done)
        if "siesta: Final" not in content and "End of run" not in content:
            return False

        # Check if geometry converged
        if self._check_geometry_converged(content):
            return False  # Already converged, no error

        # Check max force
        max_force = self._get_max_force(content)
        if max_force is None:
            return False  # Can't determine force, assume OK

        # Geometry not converged and forces above tolerance
        if max_force > self.force_tolerance:
            logger.info(
                f"Geometry not converged: max_force={max_force:.4f} eV/Ang "
                f"> tolerance={self.force_tolerance:.4f} eV/Ang"
            )
            return True

        return False

    def correct(self, directory="./") -> dict:
        """Apply geometry convergence corrections.

        Corrections are applied progressively based on the number
        of corrections already applied (tracked by custodian).

        Parameters
        ----------
        directory : str, optional
            Directory containing SIESTA output (default: "./")

        Returns
        -------
        dict
            Custodian format: {"errors": [...], "actions": [...]}
            Returns {"errors": [...], "actions": None} if unfixable
        """
        directory = Path(directory)
        corrections = {}

        # Use custodian's auto-tracked n_applied_corrections
        attempt = self.n_applied_corrections + 1

        # Read current MD.NumCGsteps from FDF
        fdf_file = directory / "siesta.fdf"
        current_params = read_fdf_file(fdf_file)
        current_steps = int(current_params.get("MD.NumCGsteps", 200))  # Default 200

        # Get current max force
        output_file = directory / "siesta.out"
        content = output_file.read_text()
        max_force = self._get_max_force(content)
        force_str = f"{max_force:.4f}" if max_force else "unknown"

        logger.info(
            f"Applying geometry convergence correction (attempt {attempt}/"
            f"{self.max_num_corrections})"
        )
        logger.info(f"  Current MD.NumCGsteps: {current_steps}")
        logger.info(f"  Current max force: {force_str} eV/Ang")

        # Strategy depends on current step count
        if current_steps < 100:
            # For small step counts, use absolute increases with minimums
            if attempt == 1:
                # Level 1: Add 50 steps, minimum 200
                new_steps = max(current_steps + 50, 200)
                corrections["MD.NumCGsteps"] = new_steps
                strategy = f"Increase MD.NumCGsteps from {current_steps} to {new_steps}"

            elif attempt == 2:
                # Level 2: Add 100 steps, minimum 300
                new_steps = max(current_steps + 100, 300)
                corrections["MD.NumCGsteps"] = new_steps
                strategy = f"Increase MD.NumCGsteps from {current_steps} to {new_steps}"

            elif attempt == 3:
                # Level 3: Set to 400 + try FIRE method
                new_steps = 400
                corrections["MD.NumCGsteps"] = new_steps
                corrections["MD.TypeOfRun"] = "CG"  # type: ignore[assignment]
                corrections["MD.FireQuench"] = True
                strategy = (
                    f"Increase MD.NumCGsteps to {new_steps}, "
                    "enable FIRE quenching (faster for some systems)"
                )

            elif attempt == 4:
                # Level 4: Set to 600 + relax convergence slightly
                new_steps = 600
                corrections["MD.NumCGsteps"] = new_steps
                corrections["MD.MaxForceTol"] = (
                    f"{self.force_tolerance * 1.5:.6f} eV/Ang"  # type: ignore[assignment]
                )
                strategy = (
                    f"Increase MD.NumCGsteps to {new_steps}, "
                    f"relax force tolerance to {self.force_tolerance * 1.5:.4f} eV/Ang"
                )

            else:
                # Level 5: Maximum steps + Broyden method (last resort)
                corrections["MD.NumCGsteps"] = 1000
                corrections["MD.TypeOfRun"] = "Broyden"  # type: ignore[assignment]
                corrections["MD.Broyden.History.Steps"] = 10
                strategy = (
                    "Use maximum MD.NumCGsteps=1000, "
                    "switch to Broyden method (last resort)"
                )

        # For larger step counts (>= 100), use percentage increases
        elif attempt == 1:
            # Level 1: Increase by 50%
            new_steps = int(current_steps * 1.5)
            corrections["MD.NumCGsteps"] = new_steps
            strategy = (
                f"Increase MD.NumCGsteps from {current_steps} to {new_steps} (+50%)"
            )

        elif attempt == 2:
            # Level 2: Increase by 100%
            new_steps = int(current_steps * 2.0)
            corrections["MD.NumCGsteps"] = new_steps
            strategy = (
                f"Increase MD.NumCGsteps from {current_steps} to {new_steps} (+100%)"
            )

        elif attempt == 3:
            # Level 3: Increase by 150% + try FIRE method
            new_steps = int(current_steps * 2.5)
            corrections["MD.NumCGsteps"] = new_steps
            corrections["MD.TypeOfRun"] = "CG"  # type: ignore[assignment]
            corrections["MD.FireQuench"] = True
            strategy = (
                f"Increase MD.NumCGsteps to {new_steps} (+150%), "
                "enable FIRE quenching (faster for some systems)"
            )

        elif attempt == 4:
            # Level 4: Increase by 200% + relax convergence slightly
            new_steps = int(current_steps * 3.0)
            corrections["MD.NumCGsteps"] = new_steps
            corrections["MD.MaxForceTol"] = f"{self.force_tolerance * 1.5:.6f} eV/Ang"  # type: ignore[assignment]
            strategy = (
                f"Increase MD.NumCGsteps to {new_steps} (+200%), "
                f"relax force tolerance to {self.force_tolerance * 1.5:.4f} eV/Ang"
            )

        else:
            # Level 5: Maximum steps + Broyden method (last resort)
            corrections["MD.NumCGsteps"] = 1000
            corrections["MD.TypeOfRun"] = "Broyden"  # type: ignore[assignment]
            corrections["MD.Broyden.History.Steps"] = 10
            strategy = (
                "Use maximum MD.NumCGsteps=1000, switch to Broyden method (last resort)"
            )

        logger.info(f"  Strategy: {strategy}")

        # Apply corrections to FDF file
        update_fdf_file(fdf_file, corrections)

        # Return custodian format
        return {
            "errors": [
                f"Geometry not converged (max_force={force_str} eV/Ang > "
                f"tolerance={self.force_tolerance:.4f} eV/Ang)"
            ],
            "actions": [f"Level {attempt}: {strategy}"],
        }

    def _get_max_force(self, content: str) -> float | None:
        """Extract maximum force from output.

        Parameters
        ----------
        content : str
            Output file content

        Returns
        -------
        float or None
            Maximum force in eV/Ang, or None if not found
        """
        # Look for max force in CG or relaxation output
        # SIESTA formats:
        #   "   Max    6.843994" (CG relaxation)
        #   "siesta: Max atomic force = 0.123456" (some versions)
        patterns = [
            # Match "   Max    value" format (most common in CG output)
            re.compile(r"^\s+Max\s+([0-9.]+)", re.MULTILINE),
            # Match "Max force = value" format
            re.compile(r"Max\s+force\s+=\s+([0-9.]+)", re.IGNORECASE),
            # Match "Maximum force = value" format
            re.compile(r"Maximum\s+force\s+=\s+([0-9.]+)", re.IGNORECASE),
            # Match "siesta: Max atomic force = value" format
            re.compile(
                r"siesta:\s+Max\s+(?:atomic\s+)?force\s+=\s+([0-9.]+)", re.IGNORECASE
            ),
        ]

        for pattern in patterns:
            matches = pattern.findall(content)
            if matches:
                # Return last (final) max force
                return float(matches[-1])

        return None

    def _check_geometry_converged(self, content: str) -> bool:
        """Check if geometry optimization converged.

        Parameters
        ----------
        content : str
            Output file content

        Returns
        -------
        bool
            True if geometry converged
        """
        patterns = [
            re.compile(r"siesta:\s+GEOM_CONV:\s+T", re.IGNORECASE),
            re.compile(r"Geometry optimization completed", re.IGNORECASE),
            re.compile(r"Forces converged", re.IGNORECASE),
            re.compile(r"SCF converged.*geometry converged", re.IGNORECASE | re.DOTALL),
        ]

        for pattern in patterns:
            if pattern.search(content):
                return True
        return False
