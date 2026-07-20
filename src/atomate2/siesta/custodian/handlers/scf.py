"""SCF convergence error handler."""

from __future__ import annotations

import logging
from pathlib import Path

from atomate2.siesta.custodian.errors import ErrorType, detect_error
from atomate2.siesta.custodian.fdf_utils import update_fdf_file
from atomate2.siesta.custodian.handlers.base import ErrorHandler

logger = logging.getLogger(__name__)


class SCFConvergenceHandler(ErrorHandler):
    """Handle SCF convergence failures.

    Applies progressively more aggressive corrections to achieve
    SCF convergence. All levels 3-10 include kick to break charge sloshing.

    Level 1-5 (Mixing adjustments):
    1. Reduce mixing weight to 0.05
    2. Reduce mixing to 0.01, increase history to 5
    3. Very small mixing (0.005), kick at step 40
    4. Pulay mixer, history 10, kick at step 50
    5. Broyden mixer, history 12, kick at step 60

    Level 6-10 (Advanced strategies + kick):
    6. 500 iterations + MP smearing (300 K), kick at step 80
    7. Restart from atomic DM, Pulay (0.02), kick at step 50
    8. Linear mixer (0.001), 800 iters, kick at step 100
    9. Mesh 200 Ry + Pulay + fresh DM, kick at step 60
    10. 1000 iters + FD smearing (500 K) + linear (0.0005), kick at step 150

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
        """Initialize SCFConvergenceHandler.

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
        """Check for SCF convergence errors.

        Parameters
        ----------
        directory : str, optional
            Directory containing SIESTA output (default: "./")

        Returns
        -------
        bool
            True if SCF convergence error detected, False otherwise
        """
        dir_path = Path(directory)
        errors = detect_error(dir_path)
        return any(error.error_type == ErrorType.SCF_CONVERGENCE for error in errors)

    def correct(self, directory: str = "./") -> dict:
        """Apply SCF convergence corrections.

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
        dir_path = Path(directory)
        corrections = {}

        # Use custodian's auto-tracked n_applied_corrections
        attempt = self.n_applied_corrections + 1

        logger.info(
            f"Applying SCF convergence correction (attempt {attempt}/"
            f"{self.max_num_corrections})"
        )

        if attempt == 1:
            # Level 1: Reduce mixing weight moderately
            corrections["SCF.Mixer.Weight"] = 0.05
            corrections["SCF.Mix.First"] = True
            strategy = "Reduce mixing weight to 0.05"

        elif attempt == 2:
            # Level 2: More aggressive mixing weight + increase history
            corrections["SCF.Mixer.Weight"] = 0.01
            corrections["SCF.Mixer.History"] = 5
            strategy = "Reduce mixing to 0.01, increase history to 5"

        elif attempt == 3:
            # Level 3: Very small mixing + mixing kick
            corrections["SCF.Mixer.Weight"] = 0.005
            corrections["SCF.Mixer.History"] = 8
            corrections["SCF.Mixer.Kick"] = 40
            corrections["SCF.Mixer.Kick.Weight"] = 0.01
            strategy = "Mixing 0.005, history 8, add kick at step 40"

        elif attempt == 4:
            # Level 4: Change mixer method to Pulay + kick
            corrections["SCF.Mixer.Method"] = "Pulay"  # type: ignore[assignment]
            corrections["SCF.Mixer.Variant"] = "Pulay"  # type: ignore[assignment]
            corrections["SCF.Mixer.Weight"] = 0.005
            corrections["SCF.Mixer.History"] = 10
            corrections["SCF.Mixer.Kick"] = 50
            corrections["SCF.Mixer.Kick.Weight"] = 0.01
            strategy = "Pulay mixer, history 10, kick at step 50"

        elif attempt == 5:
            # Level 5: Try Broyden mixer + kick
            corrections["SCF.Mixer.Method"] = "Broyden"  # type: ignore[assignment]
            corrections["SCF.Mixer.Variant"] = "Broyden"  # type: ignore[assignment]
            corrections["SCF.Mixer.Weight"] = 0.001
            corrections["SCF.Mixer.History"] = 12
            corrections["SCF.Mixer.Kick"] = 60
            corrections["SCF.Mixer.Kick.Weight"] = 0.005
            strategy = "Broyden mixer, history 12, kick at step 60"

        elif attempt == 6:
            # Level 6: Increase iterations + electronic temperature smearing + kick
            corrections["MaxSCFIterations"] = 500
            corrections["OccupationFunction"] = "MP"  # type: ignore[assignment]
            corrections["OccupationMPOrder"] = 2
            corrections["ElectronicTemperature"] = "300 K"  # type: ignore[assignment]
            corrections["SCF.Mixer.Weight"] = 0.01
            corrections["SCF.Mixer.Kick"] = 80
            corrections["SCF.Mixer.Kick.Weight"] = 0.02
            strategy = "500 iters, MP smearing (300 K), kick at step 80"

        elif attempt == 7:
            # Level 7: Restart from atomic densities + kick
            corrections["DM.Init"] = "atomic"  # type: ignore[assignment]
            corrections["DM.UseSaveDM"] = False
            corrections["SCF.Mixer.Method"] = "Pulay"  # type: ignore[assignment]
            corrections["SCF.Mixer.Weight"] = 0.02
            corrections["SCF.Mixer.History"] = 8
            corrections["SCF.Mixer.Kick"] = 50
            corrections["SCF.Mixer.Kick.Weight"] = 0.02
            strategy = "Atomic DM, Pulay (0.02), kick at step 50"

        elif attempt == 8:
            # Level 8: Ultra-conservative linear mixer + kick
            corrections["SCF.Mixer.Method"] = "Linear"  # type: ignore[assignment]
            corrections["SCF.Mixer.Weight"] = 0.001
            corrections["MaxSCFIterations"] = 800
            corrections["DM.Init"] = "atomic"  # type: ignore[assignment]
            corrections["DM.UseSaveDM"] = False
            corrections["SCF.Mixer.Kick"] = 100
            corrections["SCF.Mixer.Kick.Weight"] = 0.01
            strategy = "Linear (0.001), 800 iters, fresh DM, kick at step 100"

        elif attempt == 9:
            # Level 9: Reduce mesh cutoff for numerical stability + kick
            corrections["Mesh.Cutoff"] = "200 Ry"  # type: ignore[assignment]
            corrections["SCF.Mixer.Method"] = "Pulay"  # type: ignore[assignment]
            corrections["SCF.Mixer.Weight"] = 0.01
            corrections["SCF.Mixer.History"] = 6
            corrections["DM.Init"] = "atomic"  # type: ignore[assignment]
            corrections["SCF.Mixer.Kick"] = 60
            corrections["SCF.Mixer.Kick.Weight"] = 0.02
            strategy = "Mesh 200 Ry, Pulay, fresh DM, kick at step 60"

        else:
            # Level 10: Maximum effort - last resort + kick
            corrections["MaxSCFIterations"] = 1000
            corrections["OccupationFunction"] = "FD"  # type: ignore[assignment]
            corrections["ElectronicTemperature"] = "500 K"  # type: ignore[assignment]
            corrections["SCF.Mixer.Method"] = "Linear"  # type: ignore[assignment]
            corrections["SCF.Mixer.Weight"] = 0.0005
            corrections["DM.Init"] = "atomic"  # type: ignore[assignment]
            corrections["DM.UseSaveDM"] = False
            corrections["SCF.Mixer.Kick"] = 150
            corrections["SCF.Mixer.Kick.Weight"] = 0.005
            strategy = "1000 iters, FD (500 K), linear (0.0005), kick at step 150"

        logger.info(f"  Strategy: {strategy}")

        # Apply corrections to FDF file
        fdf_file = dir_path / "siesta.fdf"
        update_fdf_file(fdf_file, corrections)

        # Return custodian format
        return {
            "errors": ["SCF did not converge within maximum iterations"],
            "actions": [f"Level {attempt}: {strategy}"],
        }
