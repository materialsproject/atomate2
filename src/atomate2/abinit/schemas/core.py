"""Core definitions of Abinit calculations documents."""

from pathlib import Path
from typing import Type, TypeVar, Union

from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure

from atomate2.abinit.sets.base import AbinitInputSet
from atomate2.common.schemas.structure import StructureMetadata

_T = TypeVar("_T", bound="AbinitTaskDocument")


class JobMetadata(BaseModel):
    """Definition of job metadata fields."""

    dir_name: str = Field(None, description="The directory of this job.")
    calc_type: str = Field(None, description="The type of calculation of this job.")


class AbinitJobSummary(JobMetadata):
    """Definition of summary information about an Abinit Job."""

    # restart_info: RestartInfo = Field(
    #     None, description="Restart information for the next job."
    # )
    # history: JobHistory = Field(None, description="Job history.")
    abinit_input_set: AbinitInputSet = Field(
        None, description="AbinitInputSet object used to perform calculation."
    )
    structure: Structure = Field(
        None, description="Final structure of the calculation."
    )
    energy: float = Field(None, description="Final energy of the calculation.")


class AbinitTaskDocument(StructureMetadata):
    """Definition of task document about an Abinit Job."""

    @classmethod
    def from_directory(
        cls: Type[_T],
        dir_name: Union[Path, str],
    ):
        """Build AbinitTaskDocument from directory."""
