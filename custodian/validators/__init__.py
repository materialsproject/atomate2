"""Output validators for SIESTA calculations.

This package provides validators to check if SIESTA calculations
completed successfully and produced valid output.
"""

from atomate2.siesta.custodian.validators.bandstructure import BandStructureValidator
from atomate2.siesta.custodian.validators.base import OutputValidator
from atomate2.siesta.custodian.validators.relaxation import RelaxationValidator
from atomate2.siesta.custodian.validators.siesta import SiestaOutputValidator


def get_validator(
    calculation_type: str = "static",
    **kwargs,
) -> OutputValidator:
    """Get appropriate validator for calculation type.

    Parameters
    ----------
    calculation_type : str
        Type of calculation ('static', 'relax', 'band_structure')
    **kwargs
        Additional arguments for validator

    Returns
    -------
    OutputValidator
        Appropriate validator instance

    Example
    -------
    >>> validator = get_validator("relax", force_tolerance=0.02)
    >>> if validator.validate("job_001"):
    ...     print("Valid output!")
    """
    validators = {
        "static": SiestaOutputValidator,
        "relax": RelaxationValidator,
        "relaxation": RelaxationValidator,
        "band_structure": BandStructureValidator,
        "bands": BandStructureValidator,
    }

    validator_class = validators.get(calculation_type.lower(), SiestaOutputValidator)

    # Filter out strict_convergence for non-relaxation validators
    # Only RelaxationValidator supports this parameter
    if validator_class != RelaxationValidator and "strict_convergence" in kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k != "strict_convergence"}

    return validator_class(**kwargs)


__all__ = [
    "OutputValidator",
    "SiestaOutputValidator",
    "RelaxationValidator",
    "BandStructureValidator",
    "get_validator",
]
