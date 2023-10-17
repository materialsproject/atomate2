from pydantic import BaseModel, Field
from pymatgen.io.openmm.sets import OpenMMSet
from atomate2.openmm.schemas.physical_state import PhysicalState
from atomate2.openmm.schemas.dcd_reports import DCDReports
from atomate2.openmm.schemas.state_reports import StateReports
from typing import Union
import pathlib


class CalculationOutput(BaseModel):
    input_set: OpenMMSet = Field(None, description="Input set for the calculation")
    physical_state: PhysicalState = Field(None, description="Physical state for the calculation")
    state_reports: StateReports = Field(None, description="State reporter output")
    dcd_reports: DCDReports = Field(None, description="DCD reporter output")

    @classmethod
    def from_directory(cls, output_dir: Union[str, pathlib.Path]):
        # need to write final input_set to output_dir
        # parse state reporter
        # will need to figure out location of dcd_reporter from  additional store, which is global

        # in this approach, we will need to write out the final input_set to the output_dir
        # should we put the OpenMMSet in a sub-directory or just dump it's
        # contents into the output_dir? that will influence this method
        output_dir = pathlib.Path(output_dir)
        input_set = OpenMMSet.from_directory(output_dir)
        return CalculationOutput(
            input_set=input_set,
            physical_state=PhysicalState.from_input_set(input_set),
            # these need to be named consistently when they are written out
            state_reporter=StateReports.from_state_file(output_dir / "state_csv"),
            dcd_reporter=DCDReports.from_dcd_file(output_dir / "trajectory_dcd"),
        )