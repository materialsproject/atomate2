from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos

logger = logging.getLogger(__name__)

__all__ = ["PhononBSDOSDoc"]


class PhononBSDOSDoc(BaseModel):
    """
    Phonon band structures and density of states data.
    """

    structure: Structure = Field(
        None,
        description="Structure of Materials Project.",
    )

    ph_bs: PhononBandStructureSymmLine = Field(
        None,
        description="Phonon band structure object.",
    )

    ph_dos: PhononDos = Field(
        None,
        description="Phonon density of states object.",
    )

    free_energy_dict: dict = Field(None, description="dict that includes temperatures and free energies")

    imaginary_modes: bool = Field(None, description="if true, structure has imaginay modes")
    # TODO: add imaginary modes?

    # TODO: add thermal properties?
