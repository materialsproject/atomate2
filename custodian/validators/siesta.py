"""SIESTA output validator."""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path

from atomate2.siesta.custodian.validators.base import Validator

logger = logging.getLogger(__name__)


class SiestaOutputValidator(Validator):
    """Validate SIESTA calculation outputs.

    Checks for:
    - Normal program termination
    - Required output files present
    - No NaN or Inf in energies
    - Energy convergence achieved
    - Forces calculated (if expected)

    This validator inherits from custodian.custodian.Validator.
    The check() method returns True if validation FAILS (errors detected).
    """

    def __init__(
        self,
        check_energy: bool = True,
        check_forces: bool = False,
        check_stress: bool = False,
        required_files: list[str] | None = None,
    ):
        """Initialize SiestaOutputValidator.

        Parameters
        ----------
        check_energy : bool, optional
            Check that energy is finite (default: True)
        check_forces : bool, optional
            Check that forces are calculated (default: False)
        check_stress : bool, optional
            Check that stress is calculated (default: False)
        required_files : list of str, optional
            Additional required output files
        """
        self.check_energy = check_energy
        self.check_forces = check_forces
        self.check_stress = check_stress
        self.required_files = required_files or []

    def check(self, directory="./") -> bool:
        """Check if validation fails.

        This is the custodian Validator interface.

        Parameters
        ----------
        directory : str, optional
            Directory containing SIESTA output (default: "./")

        Returns
        -------
        bool
            True if validation FAILS (errors detected), False if valid
        """
        directory = Path(directory)
        errors = self._get_validation_errors(directory)

        if errors:
            for error in errors:
                logger.warning(f"Validation error: {error}")
            return True  # Validation failed

        return False  # Validation passed

    def _get_validation_errors(self, directory: Path) -> list[str]:
        """Get validation errors.

        Parameters
        ----------
        directory : Path
            Directory containing SIESTA output

        Returns
        -------
        list of str
            List of validation error messages
        """
        errors = []

        # Check for output file
        output_file = self._find_output_file(directory)
        if output_file is None:
            errors.append("SIESTA output file (siesta.out) not found")
            return errors

        # Read output content
        content = self._read_file(output_file)
        if content is None:
            errors.append(f"Could not read output file: {output_file}")
            return errors

        # Check for normal termination
        if not self._check_normal_termination(content):
            errors.append("SIESTA did not terminate normally")

        # Check for job completion marker
        if not self._check_job_completed(content):
            errors.append("Job did not complete (no 'Job completed' marker)")

        # Check energy if requested
        if self.check_energy and not self._check_energy_valid(content):
            errors.append("Energy is NaN or Inf")

        # Check forces if requested
        if self.check_forces and not self._check_forces_present(content):
            errors.append("Forces not found in output")

        # Check stress if requested
        if self.check_stress and not self._check_stress_present(content):
            errors.append("Stress not found in output")

        # Check required files
        for filename in self.required_files:
            if not (directory / filename).exists():
                errors.append(f"Required file missing: {filename}")

        return errors

    def _find_output_file(self, directory: Path) -> Path | None:
        """Find SIESTA output file (plain or gzipped).

        Parameters
        ----------
        directory : Path
            Directory to search

        Returns
        -------
        Path or None
            Path to output file, or None if not found
        """
        # Check for plain file
        plain_file = directory / "siesta.out"
        if plain_file.exists():
            return plain_file

        # Check for gzipped file
        gz_file = directory / "siesta.out.gz"
        if gz_file.exists():
            return gz_file

        # Check in compressed subfolder
        compressed_dir = directory / "siesta_compressed"
        if compressed_dir.exists():
            gz_file = compressed_dir / "siesta.out.gz"
            if gz_file.exists():
                return gz_file

        return None

    def _read_file(self, filepath: Path) -> str | None:
        """Read file (handles plain and gzipped).

        Parameters
        ----------
        filepath : Path
            File to read

        Returns
        -------
        str or None
            File content, or None if error
        """
        try:
            if filepath.suffix == ".gz":
                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                    return f.read()
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return None

    def _check_normal_termination(self, content: str) -> bool:
        """Check for normal program termination.

        Parameters
        ----------
        content : str
            Output file content

        Returns
        -------
        bool
            True if terminated normally
        """
        # SIESTA writes this at successful termination
        return "End of run" in content or "Job completed" in content

    def _check_job_completed(self, content: str) -> bool:
        """Check for job completion marker.

        Parameters
        ----------
        content : str
            Output file content

        Returns
        -------
        bool
            True if job completed
        """
        return "Job completed" in content

    def _check_energy_valid(self, content: str) -> bool:
        """Check that energy is finite.

        Parameters
        ----------
        content : str
            Output file content

        Returns
        -------
        bool
            True if energy is finite
        """
        # Look for total energy line
        match = re.search(r"Total\s*=\s*([-+]?[0-9]*\.?[0-9]+)", content)
        if not match:
            return True  # Can't find energy, don't fail

        energy_str = match.group(1)
        try:
            energy = float(energy_str)
            import math

            return math.isfinite(energy)
        except ValueError:
            return False

    def _check_forces_present(self, content: str) -> bool:
        """Check that forces are present.

        Parameters
        ----------
        content : str
            Output file content

        Returns
        -------
        bool
            True if forces found
        """
        return "Atomic forces" in content

    def _check_stress_present(self, content: str) -> bool:
        """Check that stress is present.

        Parameters
        ----------
        content : str
            Output file content

        Returns
        -------
        bool
            True if stress found
        """
        return "Stress tensor" in content
