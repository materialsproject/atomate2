from typing import Optional, List, Union
from pathlib import Path

from pydantic import BaseModel, Field

from emmet.core.vasp.task_valid import TaskState

import pandas as pd


class CalculationInput(BaseModel, extra="allow"):

    steps: int = Field(0, description="Total steps")

    step_size: int = Field(None, description="")

    platform_name: str = Field(None, description="Platform name")

    platform_properties: dict = Field(None, description="Platform properties")

    state_interval: int = Field(None, description="")

    dcd_interval: int = Field(None, description="Report interval")

    wrap_dcd: bool = Field(None, description="Wrap particles or not")

    temperature: float = Field(
        None, description="Final temperature for the calculation"
    )

    pressure: float = Field(None, description="Pressure for the calculation")

    friction_coefficient: float = Field(
        None, description="Friction coefficient for the calculation"
    )

    # integrator: Optional[str] = Field(None, description="Total steps")


class CalculationOutput(BaseModel):

    dir_name: str = Field(None, description="The directory for this OpenMM task")

    steps: List[int] = Field(None, description="List of steps")

    time: List[float] = Field(None, description="List of times")

    potential_energy: List[float] = Field(
        None, description="List of potential energies"
    )

    kinetic_energy: List[float] = Field(None, description="List of kinetic energies")

    total_energy: List[float] = Field(None, description="List of total energies")

    temperature: List[float] = Field(None, description="List of temperatures")

    volume: List[float] = Field(None, description="List of volumes")

    density: List[float] = Field(None, description="List of densities")

    elapsed_time: float = Field(None, description="Elapsed time for the calculation")

    @classmethod
    def from_directory(
        cls,
        dir_name: Path | str,
        elapsed_time: Optional[float] = None,
    ):
        state_file = Path(dir_name) / "state_csv"
        data = pd.read_csv(state_file, header=0)
        column_name_map = {
            '#"Step"': "steps",
            "Potential Energy (kJ/mole)": "potential_energy",
            "Kinetic Energy (kJ/mole)": "kinetic_energy",
            "Total Energy (kJ/mole)": "total_energy",
            "Temperature (K)": "temperature",
            "Box Volume (nm^3)": "volume",
            "Density (g/mL)": "density",
        }
        data = data.rename(columns=column_name_map)
        data = data.filter(items=column_name_map.values())
        attributes = data.to_dict(orient="list")

        return CalculationInput(
            dir_name=dir_name,
            elapsed_time=elapsed_time,
            **attributes,
        )


class Calculation(BaseModel):

    dir_name: Optional[str] = Field(
        None, description="The directory for this OpenMM calculation"
    )

    has_openmm_completed: Optional[Union[TaskState, bool]] = Field(
        None, description="Whether OpenMM completed the calculation successfully"
    )

    input: Optional[CalculationInput] = Field(
        None, description="OpenMM input settings for the calculation"
    )
    output: Optional[CalculationOutput] = Field(
        None, description="The OpenMM calculation output"
    )

    completed_at: Optional[str] = Field(
        None, description="Timestamp for when the calculation was completed"
    )
    task_name: Optional[str] = Field(
        None, description="Name of task given by custodian (e.g., relax1, relax2)"
    )

    calc_type: Optional[str] = Field(
        None,
        description="Return calculation type (run type + task_type). or just new thing",
    )
