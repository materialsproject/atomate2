from pydantic import BaseModel, Field
from typing import List, Union
from atomate2.openmm.constants import OpenMMConstants
import numpy as np
import pathlib
# from atomate2.openmm.logger import logger


class StateReports(BaseModel):
    steps: List[int] = Field(None, description="List of steps")
    time: List[float] = Field(None, description="List of times")
    potential_energy: List[float] = Field(None, description="List of potential energies")
    kinetic_energy: List[float] = Field(None, description="List of kinetic energies")
    total_energy: List[float] = Field(None, description="List of total energies")
    temperature: List[float] = Field(None, description="List of temperatures")
    volume: List[float] = Field(None, description="List of volumes")
    density: List[float] = Field(None, description="List of densities")

    @classmethod
    def from_state_file(cls, state_file: Union[str, pathlib.Path]):
        data = np.loadtxt(state_file, delimiter=',', skiprows=1)

        if len(data) == 0:
            # logger.warning(f"The loaded state file: {state_file}, was empty")
            return StateReports()
        else:
            # Extract the data columns and set the corresponding class fields
            attributes = {
                attribute: data[:, i].tolist() for i, attribute in enumerate(OpenMMConstants.STATE_REPORT_SCHEMA.value)
            }
            return StateReports(**attributes)
