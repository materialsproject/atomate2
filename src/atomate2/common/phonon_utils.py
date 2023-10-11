"""Define utilities for the phonon workflow."""
from phonopy.units import VaspToTHz

from atomate2.aims.utils.units import omega_to_THz


def get_factor(code: str):
    """
    Get the frequency conversion factor to THz for each code.

    Parameters
    ----------
    code: str
        The code to get the conversion factor for

    Returns
    -------
    float
        The correct conversion factor

    Raises
    ------
    ValueError
        If code is not defined
    """
    # TARP: This is based on self.code == "vasp" in the old forcefields workflow
    if code in ["forcefields", "vasp"]:
        return VaspToTHz
    if code == "aims":
        return omega_to_THz  # Based on CODATA 2002
    raise ValueError(f"Frequency conversion factor for code ({code}) not defined.")
