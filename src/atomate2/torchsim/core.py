"""Core module for TorchSim makers in atomate2."""

from __future__ import annotations

import os
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch_sim as ts
from jobflow import Maker, Response, job
from pymatgen.core import Structure
from torch_sim.autobatching import BinningAutoBatcher, InFlightAutoBatcher

from atomate2.torchsim.schema import (
    CONVERGENCE_FN_REGISTRY,
    PROPERTY_FN_REGISTRY,
    AutobatcherDetails,
    CalculationOutput,
    ConvergenceFn,
    PropertyFn,
    TaskType,
    TorchSimCalculation,
    TorchSimModelType,
    TorchSimTaskDoc,
    TrajectoryReporterDetails,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from torch_sim.models.interface import ModelInterface
    from torch_sim.optimizers import Optimizer
    from torch_sim.trajectory import TrajectoryReporter


def torchsim_job(method: Callable) -> job:
    """Decorate the ``make`` method of TorchSim job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all TorchSim jobs. Namely, configures the output schema to be a
    :obj:`.TorchSimTaskDoc`.

    Parameters
    ----------
    method : callable
        A TorchSim maker's make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate jobs.
    """
    return job(method, output_schema=TorchSimTaskDoc)


def properties_to_calculation_output(
    all_properties_lists: list[dict[str, list]],
) -> CalculationOutput:
    """Convert properties from ts.static to a CalculationOutput.

    Parameters
    ----------
    all_properties_lists : list[dict[str, list]]
        List of property dictionaries from ts.static, with tensors converted to lists.

    Returns
    -------
    CalculationOutput
        The calculation output containing energy, forces, and stress.
    """
    # When trajectory_reporter is used, ts.static returns empty dicts

    energy = [prop_dict["potential_energy"][0] for prop_dict in all_properties_lists]
    forces = (
        [prop_dict["forces"] for prop_dict in all_properties_lists]
        if "forces" in all_properties_lists[-1]
        else None
    )
    stress = (
        [prop_dict["stress"][0] for prop_dict in all_properties_lists]
        if "stress" in all_properties_lists[-1]
        else None
    )
    return CalculationOutput(
        energies=energy, all_forces=forces or None, stress=stress or None
    )


def get_calculation_output(
    state: ts.SimState,
    model: ModelInterface,
    autobatcher: BinningAutoBatcher | InFlightAutoBatcher | bool = False,
) -> CalculationOutput:
    """Run a static calculation and return the output.

    Parameters
    ----------
    state : ts.SimState
        The simulation state to calculate properties for.
    model : ModelInterface
        The model to use for the calculation.
    autobatcher : BinningAutoBatcher | InFlightAutoBatcher | bool
        Optional autobatcher for batching calculations. If an InFlightAutoBatcher
        is passed, it will be converted to a BinningAutoBatcher.

    Returns
    -------
    CalculationOutput
        The calculation output containing energy, forces, and stress.
    """
    # Convert InFlightAutoBatcher to BinningAutoBatcher for ts.static
    if isinstance(autobatcher, InFlightAutoBatcher):
        autobatcher = BinningAutoBatcher(
            model=model,
            memory_scales_with=autobatcher.memory_scales_with,
            max_memory_scaler=autobatcher.max_memory_scaler,
        )

    properties = ts.static(system=state, model=model, autobatcher=autobatcher)

    all_properties_lists = [
        {name: t.tolist() for name, t in prop_dict.items()} for prop_dict in properties
    ]
    return properties_to_calculation_output(all_properties_lists)


def process_trajectory_reporter_dict(
    trajectory_reporter_dict: dict[str, Any] | None,
) -> tuple[TrajectoryReporter | None, TrajectoryReporterDetails | None]:
    """Process the input dict into a TrajectoryReporter and details dictionary.

    Parameters
    ----------
    trajectory_reporter_dict : dict[str, Any] | None
        Dictionary configuration for the trajectory reporter.

    Returns
    -------
    tuple[TrajectoryReporter | None, TrajectoryReporterDetails | None]
        The trajectory reporter instance and its details dictionary.
    """
    if trajectory_reporter_dict is None:
        return None, None
    trajectory_reporter_dict = deepcopy(trajectory_reporter_dict)

    prop_calculators = trajectory_reporter_dict.pop("prop_calculators", {})

    # Convert prop_calculators to PropertyFn types and get functions
    prop_calculators_typed: dict[int, list[PropertyFn]] = {
        i: [PropertyFn(prop) if isinstance(prop, str) else prop for prop in props]
        for i, props in prop_calculators.items()
    }
    prop_calculators_functions = {
        i: {prop: PROPERTY_FN_REGISTRY[prop] for prop in props}
        for i, props in prop_calculators_typed.items()
    }

    trajectory_reporter = ts.TrajectoryReporter(
        **trajectory_reporter_dict, prop_calculators=prop_calculators_functions
    )

    trajectory_reporter.filenames = [
        Path(p).resolve() for p in trajectory_reporter_dict.get("filenames", [])
    ]

    reporter_details = TrajectoryReporterDetails(
        state_frequency=trajectory_reporter.state_frequency,
        trajectory_kwargs=trajectory_reporter.trajectory_kwargs,
        prop_calculators=prop_calculators_typed,
        state_kwargs=trajectory_reporter.state_kwargs,
        metadata=trajectory_reporter.metadata,
        filenames=trajectory_reporter.filenames,
    )
    return trajectory_reporter, reporter_details


def _get_autobatcher_details(
    autobatcher: InFlightAutoBatcher | BinningAutoBatcher,
) -> AutobatcherDetails:
    """Extract the metadata of an autobatcher.

    Parameters
    ----------
    autobatcher : InFlightAutoBatcher | BinningAutoBatcher
        The autobatcher to convert.

    Returns
    -------
    AutobatcherDetails
        Dictionary representation of the autobatcher.
    """
    return AutobatcherDetails(
        autobatcher=type(autobatcher).__name__,  # type: ignore[arg-type]
        memory_scales_with=autobatcher.memory_scales_with,  # type: ignore[arg-type]
        max_memory_scaler=autobatcher.max_memory_scaler,
        max_atoms_to_try=autobatcher.max_atoms_to_try,
        memory_scaling_factor=autobatcher.memory_scaling_factor,
        max_iterations=(
            autobatcher.max_iterations
            if isinstance(autobatcher, InFlightAutoBatcher)
            else None
        ),
        max_memory_padding=autobatcher.max_memory_padding,
    )


def process_in_flight_autobatcher_dict(
    structures: list[Structure],
    model: ModelInterface,
    autobatcher_dict: dict[str, Any] | bool,
    max_iterations: int,
) -> tuple[InFlightAutoBatcher | bool, AutobatcherDetails | None]:
    """Process the input dict into a InFlightAutoBatcher and details dictionary.

    Parameters
    ----------
    structures : list[Structure]
        List of pymatgen Structures.
    model : ModelInterface
        The model interface.
    autobatcher_dict : dict[str, Any] | bool
        Dictionary configuration for the autobatcher or a boolean.
    max_iterations : int
        Maximum number of iterations.

    Returns
    -------
    tuple[InFlightAutoBatcher | bool, AutobatcherDetails | None]
        The autobatcher instance (or False) and its details dictionary.
    """
    if isinstance(autobatcher_dict, bool):
        # False means no autobatcher
        if not autobatcher_dict:
            return False, None
        # otherwise, configure the autobatcher, with the private runners method
        state = ts.initialize_state(structures, model.device, model.dtype)
        autobatcher = ts.runners._configure_in_flight_autobatcher(  # noqa: SLF001
            state, model, autobatcher=autobatcher_dict, max_iterations=max_iterations
        )
    else:
        autobatcher_dict.setdefault("memory_scales_with", model.memory_scales_with)
        autobatcher = InFlightAutoBatcher(model=model, **autobatcher_dict)

    autobatcher_details = _get_autobatcher_details(autobatcher)
    return autobatcher, autobatcher_details


def process_binning_autobatcher_dict(
    structures: list[Structure],
    model: ModelInterface,
    autobatcher_dict: dict[str, Any] | bool,
) -> tuple[BinningAutoBatcher | bool, AutobatcherDetails | None]:
    """Process the input dict into a BinningAutoBatcher and details dictionary.

    Parameters
    ----------
    structures : list[Structure]
        List of pymatgen Structures.
    model : ModelInterface
        The model interface.
    autobatcher_dict : dict[str, Any] | bool
        Dictionary configuration for the autobatcher or a boolean.

    Returns
    -------
    tuple[BinningAutoBatcher | bool, AutobatcherDetails | None]
        The autobatcher instance (or False) and its details dictionary.
    """
    if isinstance(autobatcher_dict, bool):
        # otherwise, configure the autobatcher, with the private runners method
        state = ts.initialize_state(structures, model.device, model.dtype)
        autobatcher = ts.runners._configure_batches_iterator(  # noqa: SLF001
            state, model, autobatcher=autobatcher_dict
        )
        # list means no autobatcher
        if isinstance(autobatcher, list):
            return False, None
    else:
        # pop max_iterations if present
        autobatcher_dict = deepcopy(autobatcher_dict)
        autobatcher_dict.pop("max_iterations", None)
        autobatcher_dict.setdefault("memory_scales_with", model.memory_scales_with)
        autobatcher = BinningAutoBatcher(model=model, **autobatcher_dict)

    autobatcher_details = _get_autobatcher_details(autobatcher)
    return autobatcher, autobatcher_details


def pick_model(
    model_type: TorchSimModelType, model_path: str | Path, **model_kwargs: Any
) -> ModelInterface:
    """Pick and instantiate a model based on the model type.

    Parameters
    ----------
    model_type : TorchSimModelType
        The type of model to instantiate.
    model_path : str | Path
        Path to the model file or checkpoint.
    **model_kwargs : Any
        Additional keyword arguments to pass to the model constructor.

    Returns
    -------
    ModelInterface
        The instantiated model.

    Raises
    ------
    ValueError
        If an invalid model type is provided.
    """
    match model_type:
        case TorchSimModelType.FAIRCHEMV1:
            from torch_sim.models.fairchem_legacy import FairChemV1Model

            return FairChemV1Model(model=model_path, **model_kwargs)

        case TorchSimModelType.FAIRCHEM:
            from torch_sim.models.fairchem import FairChemModel

            return FairChemModel(model=model_path, **model_kwargs)

        case TorchSimModelType.GRAPHPESWRAPPER:
            from torch_sim.models.graphpes import GraphPESWrapper

            return GraphPESWrapper(model=model_path, **model_kwargs)

        case TorchSimModelType.MACE:
            from torch_sim.models.mace import MaceModel

            return MaceModel(model=model_path, **model_kwargs)

        case TorchSimModelType.MATTERSIM:
            from torch_sim.models.mattersim import MatterSimModel

            return MatterSimModel(model=model_path, **model_kwargs)

        case TorchSimModelType.METATOMIC:
            from torch_sim.models.metatomic import MetatomicModel

            return MetatomicModel(model=model_path, **model_kwargs)

        case TorchSimModelType.NEQUIPFRAMEWORK:
            from torch_sim.models.nequip_framework import NequIPFrameworkModel

            return NequIPFrameworkModel(model=model_path, **model_kwargs)

        case TorchSimModelType.ORB:
            from torch_sim.models.orb import OrbModel

            return OrbModel(model=model_path, **model_kwargs)

        case TorchSimModelType.SEVENNET:
            from torch_sim.models.sevennet import SevenNetModel

            return SevenNetModel(model=model_path, **model_kwargs)

        case TorchSimModelType.LENNARD_JONES:
            from torch_sim.models.lennard_jones import LennardJonesModel

            return LennardJonesModel(**model_kwargs)

        case _:
            raise ValueError(f"Invalid model type: {model_type}")


@dataclass
class TorchSimOptimizeMaker(Maker):
    """A maker class for performing geometry optimization using TorchSim.

    Parameters
    ----------
    optimizer : Optimizer
        The TorchSim optimizer to use (e.g., ts.FIRE, ts.LBFGS).
    model_type : TorchSimModelType
        The type of model to use, limited to types supported by TorchSim.
        See :obj:`.TorchSimModelType` for available options.
    model_path : str | Path
        Path to the model file or checkpoint. For some models, string names
        may be allowed (e.g., "uma-s-1" for FairChemModel).
    model_kwargs : dict[str, Any]
        Keyword arguments passed to the model constructor.
    name : str
        The name of the job.
    convergence_fn : ConvergenceFn
        The convergence function type, either "energy" or "force". This uses
        either ts.generate_energy_convergence_fn or ts.generate_force_convergence_fn
        to internally generate the convergence function. Arguments can be supplied
        via convergence_fn_kwargs. See :obj:`.CONVERGENCE_FN_REGISTRY` for options.
    convergence_fn_kwargs : dict | None
        Keyword arguments passed to the convergence function generator (e.g.,
        {"fmax": 0.01} for force convergence or {"energy_tol": 1e-6} for energy).
    trajectory_reporter_dict : dict | None
        Dictionary configuration for the trajectory reporter. Available keys:

        - ``filenames``: str | Path | list[str | Path] - Output filenames for
          trajectory data (typically .h5md files).
        - ``state_frequency``: int | None - Frequency at which states are reported.
        - ``prop_calculators``: dict[int, list[PropertyFn]] | None - Property
          calculators to apply at specific frequencies. Keys are frequencies,
          values are lists of :obj:`.PropertyFn` enums (e.g., "potential_energy",
          "forces", "stress", "kinetic_energy", "temperature", "max_force").
        - ``state_kwargs``: dict[str, Any] | None - Keyword arguments for state
          reporting.
        - ``metadata``: dict[str, str] | None - Optional metadata for the trajectory.
        - ``trajectory_kwargs``: dict[str, Any] | None - Keyword arguments for
          trajectory reporter initialization.
    autobatcher_dict : dict | bool
        Dictionary configuration for the autobatcher or a boolean. If True,
        TorchSim will automatically configure an InFlightAutoBatcher. If False,
        no autobatching is used. If a dict, available keys are:

        - ``memory_scales_with``: "n_atoms" | "n_atoms_x_density" - How memory
          usage scales with system size.
        - ``max_memory_scaler``: float | None - Maximum memory scaling factor.
        - ``max_atoms_to_try``: int | None - Maximum number of atoms to try in
          batching.
        - ``memory_scaling_factor``: float | None - Factor for memory scaling
          calculations.
        - ``max_iterations``: int | None - Maximum number of autobatching
          iterations (only used by InFlightAutoBatcher).
        - ``max_memory_padding``: float | None - Maximum padding for memory
          allocation.
    max_steps : int
        Maximum number of optimization steps to run.
    steps_between_swaps : int
        Number of steps to take before checking convergence and swapping out
        converged systems.
    init_kwargs : dict | None
        Keyword arguments passed to the optimizer initialization function.
    optimizer_kwargs : dict | None
        Keyword arguments passed to the optimizer step function.
    tags : list[str] | None
        Tags for the job.
    """

    optimizer: Optimizer
    model_type: TorchSimModelType
    model_path: str | Path
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = "torchsim optimize"
    convergence_fn: ConvergenceFn = ConvergenceFn.FORCE  # type: ignore[assignment]
    convergence_fn_kwargs: dict | None = None
    trajectory_reporter_dict: dict | None = None
    autobatcher_dict: dict | bool = False
    max_steps: int = 10_000
    steps_between_swaps: int = 5
    init_kwargs: dict | None = None
    optimizer_kwargs: dict | None = None
    tags: list[str] | None = None

    @torchsim_job
    def make(
        self,
        structure: Structure | list[Structure],
        prev_task: TorchSimTaskDoc | None = None,
        prev_dir: str | Path | None = None,
    ) -> Response:
        """Run a TorchSim optimization calculation.

        Parameters
        ----------
        structure : Structure | list[Structure]
            A pymatgen Structure or list of Structures to optimize.
        prev_task : TorchSimTaskDoc | None
            Previous task document if continuing from a previous calculation.
        prev_dir : str | Path | None
            A previous calculation directory to copy output files from. Unused, just
            added to match the method signature of other makers.

        Returns
        -------
        Response
            A response object containing the output task document.
        """
        structures = [structure] if isinstance(structure, Structure) else structure

        model = pick_model(self.model_type, self.model_path, **self.model_kwargs)

        convergence_fn_obj = CONVERGENCE_FN_REGISTRY[self.convergence_fn](
            **(self.convergence_fn_kwargs or {})
        )

        # Configure trajectory reporter
        trajectory_reporter, trajectory_reporter_details = (
            process_trajectory_reporter_dict(self.trajectory_reporter_dict)
        )

        # Configure autobatcher
        max_iterations = self.max_steps // self.steps_between_swaps
        autobatcher, autobatcher_details = process_in_flight_autobatcher_dict(
            structures,
            model,
            autobatcher_dict=self.autobatcher_dict,
            max_iterations=max_iterations,
        )

        optimizer_kwargs = self.optimizer_kwargs or {}

        start_time = time.time()
        state = ts.optimize(
            system=structures,
            model=model,
            optimizer=self.optimizer,
            convergence_fn=convergence_fn_obj,
            trajectory_reporter=trajectory_reporter,
            autobatcher=autobatcher,
            max_steps=self.max_steps,
            steps_between_swaps=self.steps_between_swaps,
            init_kwargs=self.init_kwargs,
            **optimizer_kwargs,
        )
        elapsed_time = time.time() - start_time

        final_structures = state.to_structures()

        # Get final calculation output
        calculation_output = get_calculation_output(state, model, autobatcher)

        # Create calculation object
        calculation = TorchSimCalculation(
            initial_structures=structures,
            structures=final_structures,
            output=calculation_output,
            trajectory_reporter=trajectory_reporter_details,
            autobatcher=autobatcher_details,
            model=self.model_type,
            model_path=str(Path(self.model_path).resolve()),
            task_type=TaskType.STRUCTURE_OPTIMIZATION,
            optimizer=self.optimizer,
            max_steps=self.max_steps,
            steps_between_swaps=self.steps_between_swaps,
            init_kwargs=self.init_kwargs or {},
            optimizer_kwargs=optimizer_kwargs,
        )

        # Create task document
        task_doc = TorchSimTaskDoc(
            structures=final_structures,
            calcs_reversed=(
                [calculation] + (prev_task.calcs_reversed if prev_task else [])
            ),
            time_elapsed=elapsed_time,
            uuid=str(uuid.uuid4()),
            dir_name=os.getcwd(),
        )

        return Response(output=task_doc)


@dataclass
class TorchSimIntegrateMaker(Maker):
    """A maker class for performing molecular dynamics using TorchSim.

    Parameters
    ----------
    integrator : Integrator
        The TorchSim integrator to use (e.g., ts.nvt_langevin, ts.npt_langevin).
    model_type : TorchSimModelType
        The type of model to use, limited to types supported by TorchSim.
        See :obj:`.TorchSimModelType` for available options.
    model_path : str | Path
        Path to the model file or checkpoint. For some models, string names
        may be allowed (e.g., "uma-s-1" for FairChemModel).
    n_steps : int
        Number of integration steps to perform.
    temperature : float | list[float]
        Temperature(s) for the simulation in Kelvin. Can be a single value or
        a list for temperature ramping.
    timestep : float
        Timestep for the integration in femtoseconds.
    model_kwargs : dict[str, Any]
        Keyword arguments passed to the model constructor.
    name : str
        The name of the job.
    trajectory_reporter_dict : dict | None
        Dictionary configuration for the trajectory reporter. Available keys:

        - ``filenames``: str | Path | list[str | Path] - Output filenames for
          trajectory data (typically .h5md files).
        - ``state_frequency``: int | None - Frequency at which states are reported.
        - ``prop_calculators``: dict[int, list[PropertyFn]] | None - Property
          calculators to apply at specific frequencies. Keys are frequencies,
          values are lists of :obj:`.PropertyFn` enums (e.g., "potential_energy",
          "forces", "stress", "kinetic_energy", "temperature", "max_force").
        - ``state_kwargs``: dict[str, Any] | None - Keyword arguments for state
          reporting.
        - ``metadata``: dict[str, str] | None - Optional metadata for the trajectory.
        - ``trajectory_kwargs``: dict[str, Any] | None - Keyword arguments for
          trajectory reporter initialization.
    autobatcher_dict : dict | bool
        Dictionary configuration for the autobatcher or a boolean. If True,
        TorchSim will automatically configure a BinningAutoBatcher. If False,
        no autobatching is used. If a dict, available keys are:

        - ``memory_scales_with``: "n_atoms" | "n_atoms_x_density" - How memory
          usage scales with system size.
        - ``max_memory_scaler``: float | None - Maximum memory scaling factor.
        - ``max_atoms_to_try``: int | None - Maximum number of atoms to try in
          batching.
        - ``memory_scaling_factor``: float | None - Factor for memory scaling
          calculations.
        - ``max_memory_padding``: float | None - Maximum padding for memory
          allocation.
    integrator_kwargs : dict | None
        Keyword arguments passed to the integrator step function.
    tags : list[str] | None
        Tags for the job.
    """

    integrator: Any  # Integrator type from torch_sim
    model_type: TorchSimModelType
    model_path: str | Path
    n_steps: int
    temperature: float | list[float]
    timestep: float
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = "torchsim integrate"
    trajectory_reporter_dict: dict | None = None
    autobatcher_dict: dict | bool = False
    integrator_kwargs: dict | None = None
    tags: list[str] | None = None

    @torchsim_job
    def make(
        self,
        structure: Structure | list[Structure],
        prev_task: TorchSimTaskDoc | None = None,
        prev_dir: str | Path | None = None,
    ) -> Response:
        """Run a TorchSim molecular dynamics calculation.

        Parameters
        ----------
        structure : Structure | list[Structure]
            A pymatgen Structure or list of Structures to simulate.
        prev_task : TorchSimTaskDoc | None
            Previous task document if continuing from a previous calculation.
        prev_dir : str | Path | None
            A previous calculation directory to copy output files from. Unused, just
            added to match the method signature of other makers.

        Returns
        -------
        Response
            A response object containing the output task document.
        """
        structures = [structure] if isinstance(structure, Structure) else structure

        model = pick_model(self.model_type, self.model_path, **self.model_kwargs)

        # Configure trajectory reporter
        trajectory_reporter, trajectory_reporter_details = (
            process_trajectory_reporter_dict(self.trajectory_reporter_dict)
        )

        # Configure autobatcher
        autobatcher, autobatcher_details = process_binning_autobatcher_dict(
            structures, model, autobatcher_dict=self.autobatcher_dict
        )

        integrator_kwargs = self.integrator_kwargs or {}

        start_time = time.time()
        state = ts.integrate(
            system=structures,
            model=model,
            integrator=self.integrator,
            n_steps=self.n_steps,
            temperature=self.temperature,
            timestep=self.timestep,
            trajectory_reporter=trajectory_reporter,
            autobatcher=autobatcher,
            **integrator_kwargs,
        )
        elapsed_time = time.time() - start_time

        # run a static calc to get energies and forces
        calculation_output = get_calculation_output(state, model, autobatcher)

        final_structures = state.to_structures()

        # Create calculation object
        calculation = TorchSimCalculation(
            initial_structures=structures,
            structures=final_structures,
            output=calculation_output,
            trajectory_reporter=trajectory_reporter_details,
            autobatcher=autobatcher_details,
            model=self.model_type,
            model_path=str(Path(self.model_path).resolve()),
            task_type=TaskType.MOLECULAR_DYNAMICS,
            integrator=self.integrator,
            n_steps=self.n_steps,
            temperature=self.temperature,
            timestep=self.timestep,
            integrator_kwargs=integrator_kwargs,
        )

        # Create task document
        task_doc = TorchSimTaskDoc(
            structures=final_structures,
            calcs_reversed=(
                [calculation] + (prev_task.calcs_reversed if prev_task else [])
            ),
            time_elapsed=elapsed_time,
            uuid=str(uuid.uuid4()),
            dir_name=os.getcwd(),
        )

        return Response(output=task_doc)


@dataclass
class TorchSimStaticMaker(Maker):
    """A maker class for performing static (single-point) calculations using TorchSim.

    This maker calculates energy, forces, and stress for a given structure or
    list of structures without performing any geometry optimization or dynamics.

    Parameters
    ----------
    model_type : TorchSimModelType
        The type of model to use, limited to types supported by TorchSim.
        See :obj:`.TorchSimModelType` for available options.
    model_path : str | Path
        Path to the model file or checkpoint. For some models, string names
        may be allowed (e.g., "uma-s-1" for FairChemModel).
    model_kwargs : dict[str, Any]
        Keyword arguments passed to the model constructor.
    name : str
        The name of the job.
    trajectory_reporter_dict : dict | None
        Dictionary configuration for the trajectory reporter. Available keys:

        - ``filenames``: str | Path | list[str | Path] - Output filenames for
          trajectory data (typically .h5md files).
        - ``state_frequency``: int | None - Frequency at which states are reported.
        - ``prop_calculators``: dict[int, list[PropertyFn]] | None - Property
          calculators to apply at specific frequencies. Keys are frequencies,
          values are lists of :obj:`.PropertyFn` enums (e.g., "potential_energy",
          "forces", "stress", "kinetic_energy", "temperature", "max_force").
        - ``state_kwargs``: dict[str, Any] | None - Keyword arguments for state
          reporting.
        - ``metadata``: dict[str, str] | None - Optional metadata for the trajectory.
        - ``trajectory_kwargs``: dict[str, Any] | None - Keyword arguments for
          trajectory reporter initialization.
    autobatcher_dict : dict | bool
        Dictionary configuration for the autobatcher or a boolean. If True,
        TorchSim will automatically configure a BinningAutoBatcher. If False,
        no autobatching is used. If a dict, available keys are:

        - ``memory_scales_with``: "n_atoms" | "n_atoms_x_density" - How memory
          usage scales with system size.
        - ``max_memory_scaler``: float | None - Maximum memory scaling factor.
        - ``max_atoms_to_try``: int | None - Maximum number of atoms to try in
          batching.
        - ``memory_scaling_factor``: float | None - Factor for memory scaling
          calculations.
        - ``max_memory_padding``: float | None - Maximum padding for memory
          allocation.
    tags : list[str] | None
        Tags for the job.
    """

    model_type: TorchSimModelType
    model_path: str | Path
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = "torchsim static"
    trajectory_reporter_dict: dict | None = None
    autobatcher_dict: dict | bool = False
    tags: list[str] | None = None

    @torchsim_job
    def make(
        self,
        structure: Structure | list[Structure],
        prev_task: TorchSimTaskDoc | None = None,
        prev_dir: str | Path | None = None,
    ) -> Response:
        """Run a TorchSim static calculation.

        Parameters
        ----------
        structure : Structure | list[Structure]
            A pymatgen Structure or list of Structures to calculate properties for.
        prev_task : TorchSimTaskDoc | None
            Previous task document if continuing from a previous calculation.
        prev_dir : str | Path | None
            A previous calculation directory to copy output files from. Unused, just
            added to match the method signature of other makers.

        Returns
        -------
        Response
            A response object containing the output task document.
        """
        structures = [structure] if isinstance(structure, Structure) else structure

        model = pick_model(self.model_type, self.model_path, **self.model_kwargs)

        # Configure trajectory reporter
        trajectory_reporter, trajectory_reporter_details = (
            process_trajectory_reporter_dict(self.trajectory_reporter_dict)
        )

        # Configure autobatcher
        autobatcher, autobatcher_details = process_binning_autobatcher_dict(
            structures, model, autobatcher_dict=self.autobatcher_dict
        )

        start_time = time.time()
        all_properties = ts.static(
            system=structures,
            model=model,
            trajectory_reporter=trajectory_reporter,
            autobatcher=autobatcher,
        )
        elapsed_time = time.time() - start_time

        # Convert tensors to lists
        all_properties_lists = [
            {name: t.tolist() for name, t in prop_dict.items()}
            for prop_dict in all_properties
        ]

        # Extract calculation output from properties
        calculation_output = properties_to_calculation_output(all_properties_lists)

        # Create calculation object
        calculation = TorchSimCalculation(
            initial_structures=structures,
            structures=structures,
            output=calculation_output,
            trajectory_reporter=trajectory_reporter_details,
            autobatcher=autobatcher_details,
            model=self.model_type,
            model_path=str(Path(self.model_path).resolve()),
            task_type=TaskType.STATIC,
            all_properties=all_properties_lists,
        )

        # Create task document
        task_doc = TorchSimTaskDoc(
            structures=structures,
            calcs_reversed=(
                [calculation] + (prev_task.calcs_reversed if prev_task else [])
            ),
            time_elapsed=elapsed_time,
            uuid=str(uuid.uuid4()),
            dir_name=os.getcwd(),
        )

        return Response(output=task_doc)
