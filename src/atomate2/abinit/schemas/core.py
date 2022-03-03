"""Core definitions of Abinit calculations documents."""

from abipy.abio.inputs import AbinitInput
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure

from atomate2.abinit.utils.common import RestartInfo
from atomate2.abinit.utils.history import JobHistory


class JobMetadata(BaseModel):
    """Definition of job metadata fields."""

    dir_name: str = Field(None, description="The directory of this job.")
    calc_type: str = Field(None, description="The type of calculation of this job.")


class AbinitJobSummary(JobMetadata):
    """Definition of summary information about an Abinit Job."""

    restart_info: RestartInfo = Field(
        None, description="Restart information for the next job."
    )
    history: JobHistory = Field(None, description="Job history.")
    abinit_input: AbinitInput = Field(
        None, description="AbinitInput object used to perform calculation."
    )
    structure: Structure = Field(
        None, description="Final structure of the calculation."
    )
