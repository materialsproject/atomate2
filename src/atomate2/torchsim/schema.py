"""Schemas for TorchSim tasks."""

from __future__ import annotations

import pathlib  # noqa: TC003
from enum import StrEnum  # type: ignore[attr-defined]
from typing import TYPE_CHECKING, Any, Literal

import torch_sim as ts
from emmet.core.math import Matrix3D, Vector3D  # noqa: TC002
from pydantic import BaseModel, Field, model_validator
from pymatgen.core import Structure  # noqa: TC002
from torch_sim.integrators import Integrator  # noqa: TC002
from torch_sim.optimizers import Optimizer  # noqa: TC002

if TYPE_CHECKING:
    from collections.abc import Callable


class TorchSimModelType(StrEnum):  # type: ignore[attr-defined]
    """Enum for model types."""

    FAIRCHEMV1 = "FairChemV1Model"
    FAIRCHEM = "FairChemModel"
    GRAPHPESWRAPPER = "GraphPESWrapper"
    MACE = "MaceModel"
    MATTERSIM = "MatterSimModel"
    METATOMIC = "MetatomicModel"
    NEQUIPFRAMEWORK = "NequIPFrameworkModel"
    ORB = "OrbModel"
    SEVENNET = "SevenNetModel"
    LENNARD_JONES = "LennardJonesModel"


class ConvergenceFn(StrEnum):  # type: ignore[attr-defined]
    """Enum for convergence function types."""

    ENERGY = "energy"
    FORCE = "force"


CONVERGENCE_FN_REGISTRY: dict[str, Callable] = {
    "energy": ts.generate_energy_convergence_fn,
    "force": ts.generate_force_convergence_fn,
}


class PropertyFn(StrEnum):
    """Registry for property calculation functions.

    Because we are not able to pass live python functions through
    workflow serialization, it is necessary to have an alternative
    mechanism. While the functions included here are quite basic,
    this gives users a place to patch in their own functions while
    maintaining compatibility.
    """

    POTENTIAL_ENERGY = "potential_energy"
    FORCES = "forces"
    STRESS = "stress"
    KINETIC_ENERGY = "kinetic_energy"
    TEMPERATURE = "temperature"


class TaskType(StrEnum):  # type: ignore[attr-defined]
    """Enum for TorchSim task types."""

    STATIC = "Static"
    STRUCTURE_OPTIMIZATION = "Structure Optimization"
    MOLECULAR_DYNAMICS = "Molecular Dynamics"


PROPERTY_FN_REGISTRY: dict[str, Callable] = {
    "potential_energy": lambda state: state.energy,
    "forces": lambda state: state.forces,
    "stress": lambda state: state.stress,
    "kinetic_energy": lambda state: ts.calc_kinetic_energy(
        velocities=state.velocities, masses=state.masses
    ),
    "temperature": lambda state: state.calc_temperature(),
}


class TrajectoryReporterDetails(BaseModel):
    """Details for a TorchSim trajectory reporter.

    Stores configuration and metadata for trajectory reporting.
    """

    state_frequency: int = Field(
        ..., description="Frequency at which states are reported."
    )

    trajectory_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=("Keyword arguments for trajectory reporter initialization."),
    )

    prop_calculators: dict[int, list[PropertyFn]] = Field(
        default_factory=dict,
        description=("Property calculators to apply at specific frequencies."),
    )

    state_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for state reporting.",
    )

    metadata: dict[str, str] | None = Field(
        None, description="Optional metadata for the trajectory reporter."
    )

    filenames: list[str | pathlib.Path] | None = Field(
        None, description="List of output filenames for trajectory data."
    )


class AutobatcherDetails(BaseModel):
    """Details for a TorchSim autobatcher configuration."""

    autobatcher: Literal["BinningAutoBatcher", "InFlightAutoBatcher"] = Field(
        ..., description="The type of autobatcher to use."
    )

    memory_scales_with: Literal["n_atoms", "n_atoms_x_density"] = Field(
        ..., description="How memory scales with system size."
    )

    max_memory_scaler: float | None = Field(
        None, description="Maximum memory scaling factor."
    )

    max_atoms_to_try: int | None = Field(
        None, description="Maximum number of atoms to try in batching."
    )

    memory_scaling_factor: float | None = Field(
        None, description="Factor for memory scaling calculations."
    )

    max_iterations: int | None = Field(
        None, description="Maximum number of autobatching iterations."
    )

    max_memory_padding: float | None = Field(
        None, description="Maximum padding for memory allocation."
    )


class CalculationOutput(BaseModel):
    """Schema for the output of a TorchSim calculation."""

    energies: list[float] = Field(..., description="Potential energy of the systems.")

    all_forces: list[list[Vector3D]] | None = Field(
        None, description="Forces on each atom in each system."
    )

    stress: list[Matrix3D] | None = Field(
        None, description="Stress tensor for each system."
    )

    @property
    def energy(self) -> float | None:
        """Return energy for the first/only structure (for phonon compatibility)."""
        if self.energies is None or len(self.energies) == 0:
            return None
        return self.energies[0]

    @property
    def forces(self) -> list[Vector3D] | None:
        """Return forces for the first/only structure (for single-structure mode)."""
        if self.all_forces is None or len(self.all_forces) == 0:
            return None
        return self.all_forces[0]


class TorchSimCalculation(BaseModel):
    """Schema for TorchSim calculation tasks.

    This schema supports three task types: Static, Structure Optimization,
    and Molecular Dynamics. Different fields are populated depending on the task_type.
    """

    # Common fields (always present)
    initial_structures: list[Structure] = Field(
        ..., description="List of initial structures for the calculation."
    )

    structures: list[Structure] = Field(
        ..., description="List of final structures from the calculation."
    )

    output: CalculationOutput = Field(
        ..., description="Output properties from the calculation."
    )

    trajectory_reporter: TrajectoryReporterDetails | None = Field(
        None, description="Configuration for the trajectory reporter."
    )

    autobatcher: AutobatcherDetails | None = Field(
        None, description="Configuration for the autobatcher."
    )

    model: TorchSimModelType = Field(
        ..., description="Name of the model used for the calculation."
    )

    model_path: str = Field(..., description="Path to the model file.")

    task_type: TaskType = Field(
        ...,
        description="Type of calculation performed (Static, Structure Optimization, "
        "or Molecular Dynamics).",
    )

    # Optimization-specific fields (populated when task_type == STRUCTURE_OPTIMIZATION)
    optimizer: Optimizer | None = Field(
        None, description="The TorchSim optimizer instance used for optimization."
    )

    max_steps: int | None = Field(
        None, description="Maximum number of optimization steps to perform."
    )

    steps_between_swaps: int | None = Field(
        None, description="Number of steps between system swaps in the optimizer."
    )

    init_kwargs: dict[str, Any] | None = Field(
        None, description="Additional keyword arguments for initialization."
    )

    optimizer_kwargs: dict[str, Any] | None = Field(
        None, description="Keyword arguments for the optimizer configuration."
    )

    # MD-specific fields (populated when task_type == MOLECULAR_DYNAMICS)
    integrator: Integrator | None = Field(
        None, description="The TorchSim integrator instance used for MD simulation."
    )

    n_steps: int | None = Field(
        None, description="Number of integration steps to perform."
    )

    temperature: float | list[float] | None = Field(
        None, description="Temperature(s) for the simulation in Kelvin."
    )

    timestep: float | None = Field(
        None, description="Timestep for the integration in femtoseconds."
    )

    integrator_kwargs: dict[str, Any] | None = Field(
        None, description="Keyword arguments for the integrator configuration."
    )

    # Static calculation-specific fields (populated when task_type == STATIC)
    all_properties: list[dict[str, list]] | None = Field(
        None, description="List of calculated properties for each structure."
    )


class TorchSimTaskDoc(BaseModel):
    """Base schema for TorchSim tasks."""

    structures: list[Structure] = Field(
        ..., description="List of final structures from the calculation."
    )

    calcs_reversed: list[TorchSimCalculation] = Field(
        ..., description="List of calculations for the task."
    )

    time_elapsed: float = Field(
        ..., description="Time elapsed for the calculation in seconds."
    )

    uuid: str = Field(..., description="Unique identifier for the task.")

    dir_name: str = Field(..., description="Directory name where the task was run.")

    # Compatibility fields for phonon workflow integration
    structure: Structure | None = Field(
        None, description="First/only final structure (for single-structure workflows)."
    )

    output: CalculationOutput | None = Field(
        None, description="Output from the most recent calculation."
    )

    @model_validator(mode="after")
    def set_compatibility_fields(self) -> TorchSimTaskDoc:
        """Set structure and output fields for workflow compatibility."""
        if self.structure is None and self.structures:
            object.__setattr__(self, "structure", self.structures[0])
        if self.output is None and self.calcs_reversed:
            object.__setattr__(self, "output", self.calcs_reversed[0].output)
        return self
