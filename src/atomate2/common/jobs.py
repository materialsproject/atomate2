"""Module defining common jobs."""

from jobflow import job
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.settings import settings


@job
def symmetrize_structure(
    structure: Structure, symprec: float = settings.SYMPREC
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
