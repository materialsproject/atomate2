"""Band structure calculation validator."""

from __future__ import annotations

from pathlib import Path

from atomate2.siesta.custodian.validators.siesta import SiestaOutputValidator


class BandStructureValidator(SiestaOutputValidator):
    """Validator for band structure calculations.

    Additional checks:
    - Band structure file (.bands) exists
    - K-points processed
    """

    def __init__(self, **kwargs) -> None:
        """Initialize BandStructureValidator."""
        super().__init__(
            required_files=["siesta.bands"],
            **kwargs,
        )

    def get_validation_errors(self, directory: Path | str) -> list[str]:
        """Get validation errors for band structure.

        Parameters
        ----------
        directory : Path or str
            Directory containing output

        Returns
        -------
        list of str
            Validation error messages
        """
        errors = super().get_validation_errors(directory)

        # Check for .bands file
        directory = Path(directory)
        if not (directory / "siesta.bands").exists():
            errors.append("Band structure file (siesta.bands) not found")

        return errors
