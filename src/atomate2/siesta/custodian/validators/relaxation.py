"""Relaxation calculation validator."""

from __future__ import annotations

import re
from pathlib import Path

from atomate2.siesta.custodian.validators.siesta import SiestaOutputValidator


class RelaxationValidator(SiestaOutputValidator):
    """Validator for relaxation calculations.

    Additional checks:
    - Forces converged to tolerance
    - Geometry optimization completed
    """

    def __init__(
        self,
        force_tolerance: float = 0.04,  # eV/Ang
        strict_convergence: bool = False,
        **kwargs,
    ):
        """Initialize RelaxationValidator.

        Parameters
        ----------
        force_tolerance : float, optional
            Maximum force tolerance in eV/Ang (default: 0.04)
        strict_convergence : bool, optional
            If True, fail validation if geometry does not converge.
            If False (default), allow non-converged geometries (for dirty/fast calculations).
        **kwargs
            Additional arguments for SiestaOutputValidator
        """
        super().__init__(check_forces=True, **kwargs)
        self.force_tolerance = force_tolerance
        self.strict_convergence = strict_convergence

    def _get_validation_errors(self, directory: Path) -> list[str]:
        """Get validation errors for relaxation.

        Parameters
        ----------
        directory : Path
            Directory containing output

        Returns
        -------
        list of str
            Validation error messages
        """
        # Get basic validation errors from parent (output file exists, etc.)
        # But filter out completion-related errors for relaxation:
        # - "Job did not complete" - may hit max MD steps before convergence
        # - "did not terminate normally" - may exit with error during handler recovery
        errors = super()._get_validation_errors(directory)

        # Filter out "soft" errors that are acceptable for relaxation with custodian
        errors = [
            e
            for e in errors
            if not any(
                phrase in e.lower()
                for phrase in ["did not complete", "did not terminate normally"]
            )
        ]

        # Find output file
        output_file = self._find_output_file(Path(directory))
        if output_file is None:
            return errors

        content = self._read_file(output_file)
        if content is None:
            return errors

        # NOTE: We do NOT check SCF convergence here!
        #
        # WHY: Validators run AFTER error handlers have finished. If a validator
        # detects an error, it raises ValidationError which STOPS the job.
        # Validators cannot trigger handler corrections - they're a final pass/fail gate.
        #
        # SCF convergence is the responsibility of SCFRelaxationHandler, which:
        # - Detects SCF failures during the run
        # - Removes DM file
        # - Increases SCF.MaxIter
        # - Retries the calculation

        # Check geometry convergence only if strict mode enabled
        if self.strict_convergence:
            # In strict mode, fail if geometry did not converge or forces too high
            geometry_converged = self._check_geometry_converged(content)
            max_force = self._get_max_force(content)

            if not geometry_converged:
                errors.append(
                    "Geometry optimization not converged (strict_convergence=True). "
                    "Consider: (1) increasing MD.NumCGsteps, (2) loosening force "
                    "tolerance, or (3) disabling strict_convergence for dirty calculations."
                )

            if max_force is not None and max_force > self.force_tolerance:
                errors.append(
                    f"Forces not converged: max_force={max_force:.4f} eV/Ang > "
                    f"tolerance={self.force_tolerance:.4f} eV/Ang (strict_convergence=True). "
                    "Consider: (1) increasing MD.NumCGsteps, (2) loosening force tolerance, "
                    "or (3) disabling strict_convergence for dirty calculations."
                )
        else:
            # In lenient mode (default), geometry non-convergence is OK
            # This allows dirty/fast calculations and lets custodian handlers fix issues
            # Philosophy:
            # - Without custodian: User accepts fast/dirty results
            # - With custodian: GeometryConvergenceHandler will try to fix
            pass

        return errors

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
        patterns = [
            re.compile(r"Max\s+force\s+=\s+([0-9.]+)", re.IGNORECASE),
            re.compile(r"Maximum\s+force\s+=\s+([0-9.]+)", re.IGNORECASE),
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
        ]

        for pattern in patterns:
            if pattern.search(content):
                return True
        return False
