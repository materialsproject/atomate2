"""Module defining common jobs."""

from jobflow import job
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2 import SETTINGS

__all__ = ["structure_to_primitive", "structure_to_conventional"]


@job
def structure_to_primitive(structure: Structure, symprec: float = SETTINGS.SYMPREC):
    """
    Job that creates a standard primitive structure.

    Parameters
    ----------
        structure: Structure object
            input structure that will be transformed
        symprec: float
            precision to determine symmetry

    Returns
    -------
    .Structure

    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_primitive_standard_structure()


@job
def structure_to_conventional(structure: Structure, symprec: float = SETTINGS.SYMPREC):
    """
    Job hat creates a standard conventional structure.

    Parameters
    ----------
    structure: Structure object
        input structure that will be transformed
    symprec: float
        precision to determine symmetry

    Returns
    -------
    .Structure

    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_conventional_standard_structure()
