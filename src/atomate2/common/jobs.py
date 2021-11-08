"""Module defining common jobs."""

from jobflow import job
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2 import SETTINGS


@job
def symmetrize_structure(
    structure: Structure, symprec: float = SETTINGS.SYMPREC
) -> Structure:
    """
    Symmetrize a structure.

    Parameters
    ----------
    structure
        A structure.
    symprec
        The symmetry precision.

    Returns
    -------
    Structure
        A symmetrized structure
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_primitive_standard_structure()
