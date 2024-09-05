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

    configuration_number: Optional[int] = Field(
        None, description="Configuration number of the expansion of the elastic tensor."
    )

    adsorption_energy: Optional[float] = Field(
        None, description="Fitted elastic tensor."
    )

    job_dir: Optional[str] = Field(
        None, description="The directories where the adsorption jobs were run."
    )

    @classmethod
    def from_adsorption(
        cls,
        structures: list[Structure],
        configuration_numbers: list[int],
        adsorption_energies: list[float],
        job_dirs: list[str],
    ) -> list[Self]:
        """Create a list of adsorption documents from lists of structures and energies.

        Parameters
        ----------
        structures : list[Structure]
            The list of structures for which strains and stresses were calculated.
        adsorption_energies : list[float]
            A list of adsorption energies.
        configuration_numbers : list[int]
            List of configuration numbers.
        job_dirs : list[str]
            List of job directories.
        """
        return [
            cls(
                structure=structure,
                configuration_number=configuration_number,
                adsorption_energy=adsorption_energy,
                job_dirs=job_dir,
            )
            for structure, configuration_number, adsorption_energy, job_dir in zip(
                structures, configuration_numbers, adsorption_energies, job_dirs
            )
        ]
