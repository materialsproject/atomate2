from pydantic import BaseModel, Field
from atomate2.utils.datetime import datetime_str
from atomate2.openmm.schemas.calculation_input import CalculationInput
from atomate2.openmm.schemas.calculation_output import CalculationOutput
from atomate2.openmm.schemas.task_details import TaskDetails
from typing import List


class OpenMMTaskDocument(BaseModel):
    """Definition of the OpenMM task document."""

    output_dir: str = Field(None, description="The directory for this OpenMM task")
    calculation_input: CalculationInput = Field(None, description="Input for the calculation")
    calculation_output: CalculationOutput = Field(None, description="Output for the calculation")
    task_details: TaskDetails = Field(None, description="Details about the task")
    elapsed_time: float = Field(None, description="Elapsed time for the calculation")
    last_updated: str = Field(default_factory=datetime_str, description="Timestamp for this task document was last updated")
    completed_at: str = Field(default=None, description="Timestamp for when this task was completed")
    task_label: str = Field(None, description="A description of the task")
    tags: List[str] = Field(None, description="Metadata tags for this task document")
