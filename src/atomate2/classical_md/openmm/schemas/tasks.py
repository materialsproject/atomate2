from atomate2.classical_md.schemas import ClassicalMDTaskDocument
from atomate2.openmm.schemas.physical_state import PhysicalState
from typing import Union, Tuple, Optional, Dict
from pathlib import Path
import pathlib
from datetime import datetime

from emmet.core.vasp.task_valid import TaskState

from openff.interchange import Interchange

from pydantic import BaseModel, Field
from typing import List, Union

# class DCDReports(BaseModel):
#     location: str = Field(None,
#                           description="Location of the DCD file")  # this should be a S3 location
#     # TODO: Add host?
#     report_interval: int = Field(None, description="Report interval")
#     enforce_periodic_box: bool = Field(None, description="Wrap particles or not")
#
#     @classmethod
#     def from_dcd_file(cls, dcd_file):
#         # TODO: will somehow need to interface with the additional store?
#         return


# class PhysicalState(BaseModel):
#     box_vectors: Tuple[
#         Tuple[float, float, float],
#         Tuple[float, float, float],
#         Tuple[float, float, float],
#     ] = Field(None, description="Box vectors for the calculation")
#     temperature: float = Field(None, description="Temperature for the calculation")
#     step_size: float = Field(None, description="Step size for the calculation")
#     friction_coefficient: float = Field(
#         None, description="Friction coefficient for the calculation"
#     )
#
#     @classmethod
#     def from_input_set(cls, input_set):
#         integrator = input_set.inputs[input_set.integrator_file].get_integrator()
#         state = input_set.inputs[input_set.state_file].get_state()
#         vector_array = state.getPeriodicBoxVectors(asNumpy=True)._value
#         box_vectors = tuple(tuple(vector) for vector in vector_array)
#         temperature = integrator.getTemperature()._value  # kelvin
#         step_size = integrator.getStepSize()._value  # picoseconds
#         friction_coefficient = integrator.getFriction()._value  # 1/picoseconds
#
#         return PhysicalState(
#             box_vectors=box_vectors,
#             temperature=temperature,
#             step_size=step_size,
#             friction_coefficient=friction_coefficient,
#         )


# from atomate2.openmm.logger import logger


# class StateReports(BaseModel):
#     steps: List[int] = Field(None, description="List of steps")
#     time: List[float] = Field(None, description="List of times")
#     potential_energy: List[float] = Field(
#         None, description="List of potential energies"
#     )
#     kinetic_energy: List[float] = Field(None, description="List of kinetic energies")
#     total_energy: List[float] = Field(None, description="List of total energies")
#     temperature: List[float] = Field(None, description="List of temperatures")
#     volume: List[float] = Field(None, description="List of volumes")
#     density: List[float] = Field(None, description="List of densities")
#
#     @classmethod
#     def from_state_file(cls, state_file: Union[str, pathlib.Path]):
#         data = np.loadtxt(state_file, delimiter=",", skiprows=1)
#
#         if len(data) == 0:
#             # logger.warning(f"The loaded state file: {state_file}, was empty")
#             return StateReports()
#         else:
#             # Extract the data columns and set the corresponding class fields
#             attributes = {
#                 attribute: data[:, i].tolist()
#                 for i, attribute in enumerate(OpenMMConstants.STATE_REPORT_SCHEMA.value)
#             }
#             return StateReports(**attributes)


# class CalculationOutput(BaseModel):
#     input_set: OpenMMSet = Field(None, description="Input set for the calculation")
#     physical_state: PhysicalState = Field(
#         None, description="Physical state for the calculation"
#     )
#     state_reports: StateReports = Field(None, description="State reporter output")
#     dcd_reports: DCDReports = Field(None, description="DCD reporter output")
#
#     @classmethod
#     def from_directory(cls, output_dir: Union[str, pathlib.Path]):
#         # need to write final input_set to output_dir
#         # parse state reporter
#         # will need to figure out location of dcd_reporter from  additional store, which is global
#
#         # in this approach, we will need to write out the final input_set to the output_dir
#         # should we put the OpenMMSet in a sub-directory or just dump it's
#         # contents into the output_dir? that will influence this method
#         output_dir = pathlib.Path(output_dir)
#         input_set = OpenMMSet.from_directory(output_dir)
#         return CalculationOutput(
#             input_set=input_set,
#             physical_state=PhysicalState.from_input_set(input_set),
#             # these need to be named consistently when they are written out
#             state_reporter=StateReports.from_state_file(output_dir / "state_csv"),
#             dcd_reporter=DCDReports.from_dcd_file(output_dir / "trajectory_dcd"),
#         )


# class TaskDetails(BaseModel):
#     name: str = Field(None, description="Task name")
#     kwargs: dict = Field(None, description="Task kwargs")
#     steps: int = Field(None, description="Total steps")
#
#     @classmethod
#     def from_maker(cls, maker):
#         maker = asdict(maker)
#         return TaskDetails(
#             name=maker.get("name"),
#             kwargs=maker,
#             steps=maker.get("steps", 0),
#         )


class CalculationInput(BaseModel):

    steps: int = Field(0, description="Total steps")

    step_size: int = Field(None, description="")

    integrator: Optional[str] = Field(None, description="Total steps")

    integrator_kwargs: dict = Field(None, description="Integrator")

    additional_forces: Optional[List] = Field(None, description="Total steps")

    additional_force_kwargs: dict = Field(None, description="Additional")


class CalculationOutput(BaseModel):

    dir_name: str = Field(None, description="The directory for this OpenMM task")

    state_report_interval: int = Field(None, description="")

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

    final_box_vectors: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ] = Field(None, description="Box vectors for the calculation")

    dcd_report_interval: int = Field(None, description="Report interval")

    wrapped_box: bool = Field(None, description="Wrap particles or not")

    elapsed_time: float = Field(None, description="Elapsed time for the calculation")

    @classmethod
    def from_directory(
        cls,
        dir_name,
        state_report_interval,
        final_box_vectors,
        dcd_report_interval,
        wrapped_box,
        elapsed_time,
    ):
        return


class Calculation(BaseModel):

    dir_name: Optional[str] = Field(
        None, description="The directory for this VASP calculation"
    )
    openmm_version: Optional[str] = Field(
        None, description="VASP version used to perform the calculation"
    )
    has_openmm_completed: Optional[Union[TaskState, bool]] = Field(
        None, description="Whether VASP completed the calculation successfully"
    )

    input: Optional[CalculationInput] = Field(
        None, description="VASP input settings for the calculation"
    )
    output: Optional[CalculationOutput] = Field(
        None, description="The VASP calculation output"
    )

    completed_at: Optional[str] = Field(
        None, description="Timestamp for when the calculation was completed"
    )
    task_name: Optional[str] = Field(
        None, description="Name of task given by custodian (e.g., relax1, relax2)"
    )
    output_file_paths: Optional[Dict[str, str]] = Field(
        None,
        description="Paths (relative to dir_name) of the VASP output files "
        "associated with this calculation",
    )

    calc_type: Optional[str] = Field(
        None,
        description="Return calculation type (run type + task_type). or just new thing",
    )

    @classmethod
    def from_directory(cls, dir_name: Union[Path, str], previous_calculation=None):
        return


# class TaskDocument(BaseModel, extra="allow"):
#     """Definition of the OpenMM task document."""
#
#     tags: Union[List[str], None] = Field(
#         [], title="tag", description="Metadata tagged to a given task."
#     )
#     dir_name: Optional[str] = Field(
#         None, description="The directory for this VASP task"
#     )
#     state: Optional[TaskState] = Field(None, description="State of this calculation")
#
#     calcs_reversed: Optional[List[Calculation]] = Field(
#         None,
#         title="Calcs reversed data",
#         description="Detailed data for each VASP calculation contributing to the task document.",
#     )
#
#     interchange: Optional[Interchange] = Field(
#         None, description="Final output structure from the task"
#     )
#
#     molecule_specs: Optional[List] = Field(
#         None, description="Molecules within the box."
#     )
#
#     forcefield: Optional[str] = Field(None, description="forcefield")
#
#     water_model: Optional[str] = Field(None, description="Water Model.")
#
#     task_type: Optional[str] = Field(None, description="The type of calculation.")
#
#     task_label: Optional[str] = Field(None, description="A description of the task")
#
#     # additional_json
#
#     last_updated: Optional[datetime] = Field(
#         None,
#         description="Timestamp for the most recent calculation for this task document",
#     )
#
#     @classmethod
#     def from_directory_and_task(cls, dir_name, prev_task):
#         return
