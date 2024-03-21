"""Schemas for OpenMM tasks."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd
from emmet.core.vasp.task_valid import TaskState
from pydantic import BaseModel, Field

from atomate2.classical_md.schemas import ClassicalMDTaskDocument


class CalculationInput(BaseModel, extra="allow"):
    """OpenMM input settings for a job, these are the attributes of the OpenMMMaker."""

    steps: Optional[int] = Field(0, description="Total steps")

    step_size: Optional[float] = Field(None, description="")

    platform_name: Optional[str] = Field(None, description="Platform name")

    platform_properties: Optional[dict] = Field(None, description="Platform properties")

    state_interval: Optional[int] = Field(None, description="")

    dcd_interval: Optional[int] = Field(None, description="Report interval")

    wrap_dcd: Optional[bool] = Field(None, description="Wrap particles or not")

    temperature: Optional[float] = Field(
        None, description="Final temperature for the calculation"
    )

    pressure: Optional[float] = Field(None, description="Pressure for the calculation")

    friction_coefficient: Optional[float] = Field(
        None, description="Friction coefficient for the calculation"
    )


class CalculationOutput(BaseModel):
    """OpenMM calculation output files and extracted data."""

    dir_name: Optional[str] = Field(
        None, description="The directory for this OpenMM task"
    )

    dcd_file: Optional[str] = Field(
        None, description="Path to the DCD file relative to `dir_name`"
    )

    state_file: Optional[str] = Field(
        None, description="Path to the state file relative to `dir_name`"
    )

    output_steps: Optional[list[int]] = Field(None, description="List of steps")

    time: Optional[list[float]] = Field(None, description="List of times")

    potential_energy: Optional[list[float]] = Field(
        None, description="List of potential energies"
    )

    kinetic_energy: Optional[list[float]] = Field(
        None, description="List of kinetic energies"
    )

    total_energy: Optional[list[float]] = Field(
        None, description="List of total energies"
    )

    temperature: Optional[list[float]] = Field(None, description="List of temperatures")

    volume: Optional[list[float]] = Field(None, description="List of volumes")

    density: Optional[list[float]] = Field(None, description="List of densities")

    elapsed_time: Optional[float] = Field(
        None, description="Elapsed time for the calculation"
    )

    @classmethod
    def from_directory(
        cls,
        dir_name: Path | str,
        elapsed_time: Optional[float] = None,
        steps: int = None,
        state_interval: int = None,
    ) -> "CalculationOutput":
        """Extract data from the output files in the directory."""
        state_file = Path(dir_name) / "state_csv"
        column_name_map = {
            '#"Step"': "output_steps",
            "Potential Energy (kJ/mole)": "potential_energy",
            "Kinetic Energy (kJ/mole)": "kinetic_energy",
            "Total Energy (kJ/mole)": "total_energy",
            "Temperature (K)": "temperature",
            "Box Volume (nm^3)": "volume",
            "Density (g/mL)": "density",
        }
        state_is_not_empty = state_file.exists() and state_file.stat().st_size > 0
        state_steps = state_interval and steps and steps // state_interval or 0
        if state_is_not_empty and (state_steps > 0):
            data = pd.read_csv(state_file, header=0)
            data = data.rename(columns=column_name_map)
            data = data.filter(items=column_name_map.values())
            data = data.iloc[-state_steps:]
            attributes = data.to_dict(orient="list")
            state_file_name = state_file.name
        else:
            attributes = {name: None for name in column_name_map.values()}
            state_file_name = None

        dcd_file = Path(dir_name) / "trajectory_dcd"
        dcd_is_not_empty = dcd_file.exists() and dcd_file.stat().st_size > 0
        dcd_file_name = dcd_file.name if dcd_is_not_empty else None

        return CalculationOutput(
            dir_name=str(dir_name),
            elapsed_time=elapsed_time,
            dcd_file=dcd_file_name,
            state_file=state_file_name,
            **attributes,
        )


class Calculation(BaseModel):
    """All input and output data for an OpenMM calculation."""

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


class OpenMMTaskDocument(ClassicalMDTaskDocument):
    """Definition of the OpenMM task document."""

    calcs_reversed: Optional[list[Calculation]] = Field(
        None,
        title="Calcs reversed data",
        description="Detailed data for each OpenMM calculation contributing to the "
        "task document.",
    )
