"""SCF convergence error handler specifically for relaxation calculations."""

from __future__ import annotations

import logging
from pathlib import Path

from atomate2.siesta.custodian.errors import ErrorType, detect_error
from atomate2.siesta.custodian.fdf_utils import update_fdf_file
from atomate2.siesta.custodian.handlers.base import ErrorHandler

logger = logging.getLogger(__name__)


class SCFRelaxationHandler(ErrorHandler):
    """Handle SCF convergence failures during relaxation calculations.

    This handler is specifically designed for relaxation jobs where SCF
    convergence failures can occur at each geometry step. It applies
    more aggressive corrections than the standard SCFConvergenceHandler:

    1. Remove DM file (corrupted density matrix)
    2. Increase MaxSCFIterations progressively
    3. Adjust mixer parameters
    4. Try different mixer methods (Pulay, Broyden)
    5. Electronic temperature smearing
    6. Restart from atomic densities

    Why different from SCFConvergenceHandler?
    -----------------------------------------
    During relaxation, SCF failures are more common because:
    - Geometry changes between steps
    - Forces push atoms to unfavorable positions temporarily
    - DM file from previous geometry may not be good guess

    Solution: Start fresh by removing DM and allowing more SCF steps.

    Correction Strategy (10 levels):
    --------------------------------
    All levels 3-10 include kick to break charge sloshing oscillations.

    Level 1-5 (Basic recovery):
    1. Remove DM, 300 iters, mixing=0.05
    2. Remove DM, 500 iters, mixing=0.01, history=5
    3. Remove DM, 1000 iters, mixing=0.005, kick at step 40
    4. Remove DM, Pulay, 1000 iters, kick at step 50
    5. Remove DM, Broyden, 1500 iters, kick at step 60

    Level 6-10 (Advanced recovery + kick):
    6. Remove DM, 2000 iters, MP (300 K), kick at step 80
    7. Remove DM, DM.Init=atomic, Pulay (0.02), kick at step 50
    8. Remove DM, DM.Init=atomic, Linear (0.001), kick at step 100
    9. Remove DM, DM.Init=atomic, mesh=200 Ry, kick at step 60
    10. Remove DM, DM.Init=atomic, 3000 iters, FD (500 K), kick at step 150

    This handler inherits from custodian.custodian.ErrorHandler
    and uses custodian's automatic correction tracking.
    """

    # Class properties for custodian library
    is_monitor = False  # Check errors at end of job
    is_terminating = True  # Terminate job on error detection
    raises_runtime_error = True  # Raise if cannot recover
    max_num_corrections = 10  # Maximum correction attempts (extended from 5)
    raise_on_max = False  # Don't raise, just stop correcting

    def __init__(self, max_attempts: int = 10) -> None:
        """Initialize SCFRelaxationHandler.

        Parameters
        ----------
        max_attempts : int, optional
            Maximum correction attempts (default: 5)

        Note
        ----
        The max_attempts parameter sets the max_num_corrections
        class property for compatibility with custodian library.
        """
        # Store for MSONable serialization
        self.max_attempts = max_attempts
        # Set max_num_corrections for custodian
        self.max_num_corrections = max_attempts
        self.error_type = ErrorType.SCF_CONVERGENCE

    def check(self, directory: str = "./") -> bool:
        """Check for SCF convergence errors during relaxation.

        Parameters
        ----------
        directory : str, optional
            Directory containing SIESTA output (default: "./")

        Returns
        -------
        bool
            True if SCF convergence error detected, False otherwise
        """
        directory = Path(directory)
        errors = detect_error(directory)
        return any(error.error_type == ErrorType.SCF_CONVERGENCE for error in errors)

    def correct(self, directory: str = "./") -> dict:
        """Apply SCF convergence corrections for relaxation.

        Corrections are applied progressively based on the number
        of corrections already applied (tracked by custodian).

        This handler is more aggressive than standard SCF handler:
        - Always removes DM file (start fresh after geometry change)
        - Increases MaxSCFIterations to allow more convergence attempts
        - Uses same mixer progression as standard handler

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
        corrections: dict[str, int | float | str | bool] = {}

        # Use custodian's auto-tracked n_applied_corrections
        attempt = self.n_applied_corrections + 1

        logger.info(
            f"Applying SCF relaxation correction (attempt {attempt}/"
            f"{self.max_num_corrections})"
        )

        # ALWAYS remove DM file for relaxation (corrupted from geometry change)
        dm_file = directory / "siesta.DM"
        if dm_file.exists():
            dm_file.unlink()
            logger.info(f"  Removed DM file: {dm_file}")

        if attempt == 1:
            # Level 1: Remove DM + increase SCF steps + reduce mixing
            corrections["MaxSCFIterations"] = 300
            corrections["SCF.Mixer.Weight"] = 0.05
            corrections["SCF.Mix.First"] = True
            strategy = (
                "Remove DM, increase MaxSCFIterations to 300, reduce mixing to 0.05"
            )

        elif attempt == 2:
            # Level 2: More SCF steps + aggressive mixing
            corrections["MaxSCFIterations"] = 500
            corrections["SCF.Mixer.Weight"] = 0.01
            corrections["SCF.Mixer.History"] = 5
            strategy = "Remove DM, MaxSCFIterations=500, mixing=0.01, history=5"

        elif attempt == 3:
            # Level 3: Many SCF steps + very small mixing + kick
            corrections["MaxSCFIterations"] = 1000
            corrections["SCF.Mixer.Weight"] = 0.005
            corrections["SCF.Mixer.History"] = 8
            corrections["SCF.Mixer.Kick"] = 40
            corrections["SCF.Mixer.Kick.Weight"] = 0.01
            strategy = (
                "Remove DM, MaxSCFIterations=1000, mixing=0.005, history=8, kick at 40"
            )

        elif attempt == 4:
            # Level 4: Switch to Pulay mixer + kick
            corrections["MaxSCFIterations"] = 1000
            corrections["SCF.Mixer.Method"] = "Pulay"
            corrections["SCF.Mixer.Variant"] = "Pulay"
            corrections["SCF.Mixer.Weight"] = 0.005
            corrections["SCF.Mixer.History"] = 10
            corrections["SCF.Mixer.Kick"] = 50
            corrections["SCF.Mixer.Kick.Weight"] = 0.01
            strategy = "Remove DM, Pulay, 1000 iters, kick at step 50"

        elif attempt == 5:
            # Level 5: Broyden mixer + kick
            corrections["MaxSCFIterations"] = 1500
            corrections["SCF.Mixer.Method"] = "Broyden"
            corrections["SCF.Mixer.Variant"] = "Broyden"
            corrections["SCF.Mixer.Weight"] = 0.001
            corrections["SCF.Mixer.History"] = 12
            corrections["SCF.Mixer.Kick"] = 60
            corrections["SCF.Mixer.Kick.Weight"] = 0.005
            strategy = "Remove DM, Broyden, 1500 iters, kick at step 60"

        elif attempt == 6:
            # Level 6: More iterations + electronic temperature smearing + kick
            corrections["MaxSCFIterations"] = 2000
            corrections["OccupationFunction"] = "MP"
            corrections["OccupationMPOrder"] = 2
            corrections["ElectronicTemperature"] = "300 K"
            corrections["SCF.Mixer.Method"] = "Pulay"
            corrections["SCF.Mixer.Weight"] = 0.01
            corrections["SCF.Mixer.History"] = 8
            corrections["SCF.Mixer.Kick"] = 80
            corrections["SCF.Mixer.Kick.Weight"] = 0.02
            strategy = "Remove DM, 2000 iters, MP (300 K), Pulay, kick at step 80"

        elif attempt == 7:
            # Level 7: Restart from atomic densities + kick
            corrections["DM.Init"] = "atomic"
            corrections["MaxSCFIterations"] = 2000
            corrections["SCF.Mixer.Method"] = "Pulay"
            corrections["SCF.Mixer.Weight"] = 0.02
            corrections["SCF.Mixer.History"] = 8
            corrections["SCF.Mixer.Kick"] = 50
            corrections["SCF.Mixer.Kick.Weight"] = 0.02
            strategy = "Remove DM, DM.Init=atomic, Pulay (0.02), kick at step 50"

        elif attempt == 8:
            # Level 8: Ultra-conservative linear mixer + kick
            corrections["DM.Init"] = "atomic"
            corrections["MaxSCFIterations"] = 2500
            corrections["SCF.Mixer.Method"] = "Linear"
            corrections["SCF.Mixer.Weight"] = 0.001
            corrections["SCF.Mixer.Kick"] = 100
            corrections["SCF.Mixer.Kick.Weight"] = 0.01
            strategy = "Remove DM, DM.Init=atomic, Linear (0.001), kick at step 100"

        elif attempt == 9:
            # Level 9: Reduce mesh cutoff for numerical stability + kick
            corrections["DM.Init"] = "atomic"
            corrections["Mesh.Cutoff"] = "200 Ry"
            corrections["MaxSCFIterations"] = 2500
            corrections["SCF.Mixer.Method"] = "Pulay"
            corrections["SCF.Mixer.Weight"] = 0.01
            corrections["SCF.Mixer.History"] = 6
            corrections["SCF.Mixer.Kick"] = 60
            corrections["SCF.Mixer.Kick.Weight"] = 0.02
            strategy = "Remove DM, DM.Init=atomic, mesh=200 Ry, kick at step 60"

        else:
            # Level 10: Maximum effort - last resort + kick
            corrections["DM.Init"] = "atomic"
            corrections["MaxSCFIterations"] = 3000
            corrections["OccupationFunction"] = "FD"
            corrections["ElectronicTemperature"] = "500 K"
            corrections["SCF.Mixer.Method"] = "Linear"
            corrections["SCF.Mixer.Weight"] = 0.0005
            corrections["SCF.Mixer.Kick"] = 150
            corrections["SCF.Mixer.Kick.Weight"] = 0.005
            strategy = "Remove DM, DM.Init=atomic, 3000 iters, FD (500 K), kick at 150"

        logger.info(f"  Strategy: {strategy}")

        # Apply corrections to FDF file
        fdf_file = directory / "siesta.fdf"
        update_fdf_file(fdf_file, corrections)

        # Return custodian format
        return {
            "errors": ["SCF did not converge during relaxation step"],
            "actions": [f"Level {attempt}: {strategy}"],
        }
