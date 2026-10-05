"""Schemas for the CALPHAD workflow outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pymatgen.core import Structure


class SqsCalculation(BaseModel):
    """Energy of one special quasirandom structure."""

    lattice: str = Field(description="ATAT lattice name, e.g. FCC_A1 or LIQUID.")
    folder: str = Field(description="Name of the ATAT folder of the structure.")
    energy: float = Field(
        description="Total energy of the structure, in eV. For the liquid, the "
        "mean potential energy of the production MD."
    )
    structure: Structure | None = Field(
        None, description="Relaxed structure. None for the liquid."
    )
    relaxation_strain: float | None = Field(
        None,
        description="Cell distortion of the relaxation, as ATAT checkrelax reports "
        "it. None for the liquid.",
    )
    is_force_converged: bool | None = Field(
        None, description="Whether the relaxation converged. None for the liquid."
    )
    mean_squared_displacement: float | None = Field(
        None,
        description="Mean squared displacement of the atoms over the production "
        "MD, in Angstrom^2. Only for the liquid.",
    )
    dir_name: str | None = Field(
        None, description="Folder of the relaxation or of the liquid MD."
    )


class CalphadDoc(BaseModel):
    """Output of the CALPHAD workflow."""

    elements: list[str] = Field(description="The two elements.")
    lattices: list[str] = Field(description="ATAT lattice names that were fitted.")
    level: int = Field(description="Level of the ATAT SQS database that was used.")
    terms: dict[str, list[str]] = Field(
        description="Lines of the sqs2tdb terms.in file for each lattice."
    )
    calculations: list[SqsCalculation] = Field(
        description="Energies of all special quasirandom structures."
    )
    fit_logs: dict[str, str] = Field(
        description="Output of sqs2tdb -fit for each lattice."
    )
    tdb: str = Field(description="Text of the TDB file written by sqs2tdb -tdb -oc.")
