"""Core module for TorchSim makers in atomate2."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch_sim as ts
from jobflow import Maker, Response, job
from torch_sim.autobatching import BinningAutoBatcher, InFlightAutoBatcher

from atomate2.torchsim.schema import (
    CONVERGENCE_FN_REGISTRY,
    PROPERTY_FN_REGISTRY,
    AutobatcherDetails,
    ConvergenceFn,
    PropertyFn,
    TorchSimModelType,
    TorchSimOptimizeCalculation,
    TorchSimTaskDoc,
    TrajectoryReporterDetails,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatgen.core import Structure
    from torch_sim.models.interface import ModelInterface
    from torch_sim.optimizers import Optimizer
    from torch_sim.state import SimState
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
    state: SimState,
    model: ModelInterface,
    autobatcher_dict: dict[str, Any] | bool,
    max_iterations: int,
) -> tuple[InFlightAutoBatcher | bool, AutobatcherDetails | None]:
    """Process the input dict into a InFlightAutoBatcher and details dictionary.

    Parameters
    ----------
    state : SimState
        The simulation state.
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
        autobatcher = ts.runners._configure_in_flight_autobatcher(  # noqa: SLF001
            state, model, autobatcher=autobatcher_dict, max_iterations=max_iterations
        )
    else:
        autobatcher = InFlightAutoBatcher(model=model, **autobatcher_dict)

    autobatcher_details = _get_autobatcher_details(autobatcher)
    return autobatcher, autobatcher_details


def process_binning_autobatcher_dict(
    state: SimState, model: ModelInterface, autobatcher_dict: dict[str, Any] | bool
) -> tuple[BinningAutoBatcher | bool, AutobatcherDetails | None]:
    """Process the input dict into a BinningAutoBatcher and details dictionary.

    Parameters
    ----------
    state : SimState
        The simulation state.
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
    if model_type == TorchSimModelType.FAIRCHEMV1:
        from torch_sim.models.fairchem_legacy import FairChemV1Model

        return FairChemV1Model(model=model_path, **model_kwargs)
    if model_type == TorchSimModelType.FAIRCHEM:
        from torch_sim.models.fairchem import FairChemModel

        return FairChemModel(model=model_path, **model_kwargs)
    if model_type == TorchSimModelType.GRAPHPESWRAPPER:
        from torch_sim.models.graphpes import GraphPESWrapper

        return GraphPESWrapper(model=model_path, **model_kwargs)
    if model_type == TorchSimModelType.MACE:
        from torch_sim.models.mace import MaceModel

        return MaceModel(model=model_path, **model_kwargs)
    if model_type == TorchSimModelType.MATTERSIM:
        from torch_sim.models.mattersim import MatterSimModel

        return MatterSimModel(model=model_path, **model_kwargs)
    if model_type == TorchSimModelType.METATOMIC:
        from torch_sim.models.metatomic import MetatomicModel

        return MetatomicModel(model=model_path, **model_kwargs)
    if model_type == TorchSimModelType.NEQUIPFRAMEWORK:
        from torch_sim.models.nequip_framework import NequIPFrameworkModel

        return NequIPFrameworkModel(model=model_path, **model_kwargs)
    if model_type == TorchSimModelType.ORB:
        from torch_sim.models.orb import OrbModel

        return OrbModel(model=model_path, **model_kwargs)
    if model_type == TorchSimModelType.SEVENNET:
        from torch_sim.models.sevennet import SevenNetModel

        return SevenNetModel(model=model_path, **model_kwargs)
    if model_type == TorchSimModelType.LENNARD_JONES:
        from torch_sim.models.lennard_jones import LennardJonesModel

        return LennardJonesModel(**model_kwargs)

    raise ValueError(f"Invalid model type: {model_type}")


@dataclass
class TorchSimOptimizeMaker(Maker):
    """A maker class for performing optimization using TorchSim.

    Parameters
    ----------
    name : str
        The name of the job.
    model : tuple[ModelType, str | Path]
        The model to use for optimization. A tuple of (model_type, model_path).
    optimizer : Optimizer
        The TorchSim optimizer to use.
    convergence_fn : ConvergenceFn | None
        The convergence function type to use.
    convergence_fn_kwargs : dict | None
        Keyword arguments for the convergence function.
    trajectory_reporter_dict : dict | None
        Dictionary configuration for the trajectory reporter.
    autobatcher_dict : dict | None
        Dictionary configuration for the autobatcher.
    max_steps : int
        Maximum number of optimization steps.
    steps_between_swaps : int
        Number of steps between system swaps.
    init_kwargs : dict | None
        Additional initialization keyword arguments.
    optimizer_kwargs : dict | None
        Keyword arguments for the optimizer.
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
        self, structures: list[Structure], prev_task: TorchSimTaskDoc | None = None
    ) -> Response:
        """Run a TorchSim optimization calculation.

        Parameters
        ----------
        structures : list[Structure]
            List of pymatgen Structures to optimize.
        prev_task : TorchSimTaskDoc | None
            Previous task document if continuing from a previous calculation.

        Returns
        -------
        Response
            A response object containing the output task document.
        """
        model = pick_model(self.model_type, self.model_path, **self.model_kwargs)

        convergence_fn_obj = CONVERGENCE_FN_REGISTRY[self.convergence_fn](
            **(self.convergence_fn_kwargs or {})
        )

        state = ts.initialize_state(structures, model.device, model.dtype)

        # Configure trajectory reporter
        trajectory_reporter, trajectory_reporter_details = (
            process_trajectory_reporter_dict(self.trajectory_reporter_dict)
        )

        # Configure autobatcher
        max_iterations = self.max_steps // self.steps_between_swaps
        autobatcher, autobatcher_details = process_in_flight_autobatcher_dict(
            state,
            model,
            autobatcher_dict=self.autobatcher_dict,
            max_iterations=max_iterations,
        )

        optimizer_kwargs = self.optimizer_kwargs or {}

        start_time = time.time()
        state = ts.optimize(
            system=state,
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

        # Create calculation object
        calculation = TorchSimOptimizeCalculation(
            initial_structures=structures,
            structures=final_structures,
            trajectory_reporter=trajectory_reporter_details,
            autobatcher=autobatcher_details,
            model=self.model_type,
            model_path=str(Path(self.model_path).resolve()),
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
                [calculation] + ([prev_task.calcs_reversed] if prev_task else [])
            ),
            time_elapsed=elapsed_time,
        )

        return Response(output=task_doc)


@dataclass
class TorchSimIntegrateMaker(Maker):
    """A maker class for performing molecular dynamics using TorchSim.

    Parameters
    ----------
    name : str
        The name of the job.
    model_type : TorchSimModelType
        The type of model to use.
    model_path : str | Path
        Path to the model file or checkpoint.
    integrator : Integrator
        The TorchSim integrator to use.
    n_steps : int
        Number of integration steps to perform.
    temperature : float | list[float]
        Temperature(s) for the simulation in Kelvin.
    timestep : float
        Timestep for the integration in femtoseconds.
    model_kwargs : dict[str, Any]
        Keyword arguments for the model.
    trajectory_reporter_dict : dict | None
        Dictionary configuration for the trajectory reporter.
    autobatcher_dict : dict | bool
        Dictionary configuration for the autobatcher.
    integrator_kwargs : dict | None
        Keyword arguments for the integrator.
    tags : list[str] | None
        Tags for the job.
    """

    model_type: TorchSimModelType
    model_path: str | Path
    integrator: Any  # Integrator type from torch_sim
    n_steps: int
    temperature: float | list[float]
    timestep: float
    name: str = "torchsim integrate"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    trajectory_reporter_dict: dict | None = None
    autobatcher_dict: dict | bool = False
    integrator_kwargs: dict | None = None
    tags: list[str] | None = None

    @torchsim_job
    def make(
        self, structures: list[Structure], prev_task: TorchSimTaskDoc | None = None
    ) -> Response:
        """Run a TorchSim molecular dynamics calculation.

        Parameters
        ----------
        structures : list[Structure]
            List of pymatgen Structures to simulate.
        prev_task : TorchSimTaskDoc | None
            Previous task document if continuing from a previous calculation.

        Returns
        -------
        Response
            A response object containing the output task document.
        """
        from atomate2.torchsim.schema import TorchSimIntegrateCalculation

        model = pick_model(self.model_type, self.model_path, **self.model_kwargs)

        state = ts.initialize_state(structures, model.device, model.dtype)

        # Configure trajectory reporter
        trajectory_reporter, trajectory_reporter_details = (
            process_trajectory_reporter_dict(self.trajectory_reporter_dict)
        )

        # Configure autobatcher
        autobatcher, autobatcher_details = process_binning_autobatcher_dict(
            state, model, autobatcher_dict=self.autobatcher_dict
        )

        integrator_kwargs = self.integrator_kwargs or {}

        start_time = time.time()
        state = ts.integrate(
            system=state,
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

        final_structures = state.to_structures()

        # Create calculation object
        calculation = TorchSimIntegrateCalculation(
            initial_structures=structures,
            structures=final_structures,
            trajectory_reporter=trajectory_reporter_details,
            autobatcher=autobatcher_details,
            model=self.model_type,
            model_path=str(Path(self.model_path).resolve()),
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
                [calculation] + ([prev_task.calcs_reversed] if prev_task else [])
            ),
            time_elapsed=elapsed_time,
        )

        return Response(output=task_doc)


@dataclass
class TorchSimStaticMaker(Maker):
    """A maker class for performing static calculations using TorchSim.

    Parameters
    ----------
    name : str
        The name of the job.
    model_type : TorchSimModelType
        The type of model to use.
    model_path : str | Path
        Path to the model file or checkpoint.
    model_kwargs : dict[str, Any]
        Keyword arguments for the model.
    trajectory_reporter_dict : dict | None
        Dictionary configuration for the trajectory reporter.
    autobatcher_dict : dict | bool
        Dictionary configuration for the autobatcher.
    tags : list[str] | None
        Tags for the job.
    """

    model_type: TorchSimModelType
    model_path: str | Path
    name: str = "torchsim static"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    trajectory_reporter_dict: dict | None = None
    autobatcher_dict: dict | bool = False
    tags: list[str] | None = None

    @torchsim_job
    def make(
        self, structures: list[Structure], prev_task: TorchSimTaskDoc | None = None
    ) -> Response:
        """Run a TorchSim static calculation.

        Parameters
        ----------
        structures : list[Structure]
            List of pymatgen Structures to calculate properties for.
        prev_task : TorchSimTaskDoc | None
            Previous task document if continuing from a previous calculation.

        Returns
        -------
        Response
            A response object containing the output task document.
        """
        from atomate2.torchsim.schema import TorchSimStaticCalculation

        model = pick_model(self.model_type, self.model_path, **self.model_kwargs)

        state = ts.initialize_state(structures, model.device, model.dtype)

        # Configure trajectory reporter
        trajectory_reporter, trajectory_reporter_details = (
            process_trajectory_reporter_dict(self.trajectory_reporter_dict)
        )

        # Configure autobatcher
        autobatcher, autobatcher_details = process_binning_autobatcher_dict(
            state, model, autobatcher_dict=self.autobatcher_dict
        )

        start_time = time.time()
        all_properties = ts.static(
            system=state,
            model=model,
            trajectory_reporter=trajectory_reporter,
            autobatcher=autobatcher,
        )
        elapsed_time = time.time() - start_time

        # Convert tensors to numpy arrays
        all_properties_numpy = [
            {name: t.cpu().numpy() for name, t in prop_dict.items()}
            for prop_dict in all_properties
        ]

        # Create calculation object
        calculation = TorchSimStaticCalculation(
            initial_structures=structures,
            structures=structures,
            trajectory_reporter=trajectory_reporter_details,
            autobatcher=autobatcher_details,
            model=self.model_type,
            model_path=str(Path(self.model_path).resolve()),
            all_properties=all_properties_numpy,
        )

        # Create task document
        task_doc = TorchSimTaskDoc(
            structures=structures,
            calcs_reversed=(
                [calculation] + ([prev_task.calcs_reversed] if prev_task else [])
            ),
            time_elapsed=elapsed_time,
        )

        return Response(output=task_doc)
