"""Schemas for dielectric properties."""

from typing import Self

import numpy as np
from emmet.core.math import Matrix3D
from emmet.core.polar import PiezoelectricDoc, PiezoTensor
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__all__ = ["DielectricDocument"]


class DielectricDocument(BaseModel):
    """
    Dielectric tensors and Born effective charges, and piezoelectric tensors if known.

    The dielectric tensors and Born effective charges are in the Cartesian frame of
    the structure.
    """

    structure: Structure = Field(description="The structure of the calculation.")
    epsilon_electronic: Matrix3D = Field(
        description="Electronic (high-frequency) dielectric tensor."
    )
    epsilon_ionic: Matrix3D | None = Field(
        None, description="Ionic contribution to the dielectric tensor."
    )
    epsilon_total: Matrix3D | None = Field(
        None,
        description="Static dielectric tensor, the sum of the electronic and ionic "
        "contributions.",
    )
    refractive_index: float = Field(
        description="Refractive index from the electronic dielectric tensor."
    )
    piezoelectric: PiezoelectricDoc | None = Field(
        None,
        description="Piezoelectric tensors in the IEEE orientation. None if they were "
        "not calculated or the structure is centrosymmetric.",
    )
    born: list[Matrix3D] = Field(
        description="Born effective charges for each site, in units of e."
    )

    @classmethod
    def from_tensors(
        cls,
        structure: Structure,
        epsilon_electronic: Matrix3D,
        born: list[Matrix3D],
        epsilon_ionic: Matrix3D | None = None,
        piezo_electronic: PiezoTensor | None = None,
        piezo_ionic: PiezoTensor | None = None,
    ) -> Self:
        """
        Create a dielectric document from the tensors of a calculation.

        Parameters
        ----------
        structure : .Structure
            The structure of the calculation.
        epsilon_electronic : Matrix3D
            The electronic dielectric tensor.
        born : list of Matrix3D
            The Born effective charges for each site.
        epsilon_ionic : Matrix3D or None
            The ionic contribution to the dielectric tensor.
        piezo_electronic : PiezoTensor or None
            The electronic piezoelectric tensor in the Voigt order of VASP.
        piezo_ionic : PiezoTensor or None
            The ionic piezoelectric tensor in the Voigt order of VASP.

        Returns
        -------
        DielectricDocument
            The dielectric document.
        """
        epsilon_total = None
        if epsilon_ionic is not None:
            epsilon_total = np.add(epsilon_electronic, epsilon_ionic).tolist()

        piezoelectric = None
        has_piezo = piezo_electronic is not None and piezo_ionic is not None
        if has_piezo and not SpacegroupAnalyzer(structure).is_laue():
            piezoelectric = PiezoelectricDoc.from_ionic_and_electronic(
                piezo_ionic, piezo_electronic, structure
            )

        return cls(
            structure=structure,
            epsilon_electronic=epsilon_electronic,
            epsilon_ionic=epsilon_ionic,
            epsilon_total=epsilon_total,
            refractive_index=np.sqrt(np.trace(epsilon_electronic) / 3),
            piezoelectric=piezoelectric,
            born=born,
        )
