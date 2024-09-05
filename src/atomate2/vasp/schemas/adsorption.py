"""Schemas for adsorption structures and related properties."""

from typing import Optional

from emmet.core.structure import StructureMetadata
from pydantic import Field
from pymatgen.core import Structure
from typing_extensions import Self


class AdsorptionDocument(StructureMetadata):
    """Document containing adsorption structure information and related properties."""

    structure: Optional[Structure] = Field(
        None, description="The structure for which the elastic data is calculated."
    )
    adsorption_energy: Optional[float] = Field(
        None, description="Fitted elastic tensor."
    )
    order: Optional[int] = Field(
        None, description="Order of the expansion of the elastic tensor."
    )

    @classmethod
    def from_stresses(
        cls,
        structure: Structure,
        adsorption_energy: list[float],
        order: Optional[int] = None,
    ) -> Self:
        """Create an adsorption document from structures and energies.

        Parameters
        ----------
        structure : .Structure
            The structure for which strains and stresses were calculated.
        adsorption_energy : list of adsorption energies
            A list of adsorption energies.
        order : int
            Order of the adsorption energies ranking from lowest to highest.
        """
        return cls.from_structure(
            structure=structure,
            meta_structure=structure,
            adsorption_energy=adsorption_energy,
            order=order,
        )
