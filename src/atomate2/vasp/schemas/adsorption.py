"""Schemas for adsorption structures and related properties."""

from pydantic import BaseModel, Field
from pymatgen.core import Structure


class AdsorptionDocument(BaseModel):
    """Document containing adsorption structures information and related properties."""

    structures: list[Structure] = Field(
        ..., description="List of adsorption structures."
    )

    configuration_numbers: list[int] = Field(
        ..., description="List of configuration numbers for the adsorption structures."
    )

    adsorption_energies: list[float] = Field(
        ..., description="List of adsorption energies corresponding to each structure."
    )

    job_dirs: list[str] = Field(
        ..., description="List of directories where the adsorption jobs were run."
    )

    @classmethod
    def from_adsorption(
        cls,
        structures: list[Structure],
        configuration_numbers: list[int],
        adsorption_energies: list[float],
        job_dirs: list[str],
    ) -> "AdsorptionDocument":
        """Create an AdsorptionDocument from lists of structures and energies.

        Parameters
        ----------
        structures : list[Structure]
            The list of adsorption structures.
        configuration_numbers : list[int]
            List of configuration numbers.
        adsorption_energies : list[float]
            A list of adsorption energies.
        job_dirs : list[str]
            List of job directories.

        Returns
        -------
        AdsorptionDocument
            An instance of AdsorptionDocument containing the provided data.
        """
        return cls(
            structures=structures,
            configuration_numbers=configuration_numbers,
            adsorption_energies=adsorption_energies,
            job_dirs=job_dirs,
        )
