"""Tests for TorchSim core makers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch_sim as ts
from ase.build import bulk
from jobflow import run_locally
from mace.calculators.foundations_models import download_mace_mp_checkpoint
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.torchsim.core import TSIntegrateMaker, TSOptimizeMaker, TSStaticMaker
from atomate2.torchsim.schema import ConvergenceFn, TSModelType


@pytest.fixture
def mace_model_path():
    """Download and return path to MACE model checkpoint."""
    return Path(download_mace_mp_checkpoint("small"))


@pytest.fixture
def ar_structure() -> Structure:
    """Create a face-centered cubic (FCC) Argon structure."""
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True)
    return AseAtomsAdaptor.get_structure(atoms)


@pytest.fixture
def fe_structure() -> Structure:
    """Create crystalline iron using ASE."""
    atoms = bulk("Fe", "fcc", a=5.26, cubic=True)
    return AseAtomsAdaptor.get_structure(atoms)


def test_relax_job_comprehensive(ar_structure: Structure, tmp_path) -> None:
    """Test TSOptimizeMaker with all kwargs.

    Includes trajectory reporter and autobatcher.
    """
    # Perturb the structure to make optimization meaningful
    perturbed_structure = ar_structure.copy()
    perturbed_structure.translate_sites(
        list(range(len(perturbed_structure))), [0.01, 0.01, 0.01]
    )

    n_systems = 2
    trajectory_reporter_dict = {
        "filenames": [tmp_path / f"relax_{i}.h5md" for i in range(n_systems)],
        "state_frequency": 5,
        "prop_calculators": {1: ["potential_energy"]},
    }

    # Create autobatcher
    autobatcher_dict = False

    maker = TSOptimizeMaker(
        model_type=TSModelType.LENNARD_JONES,
        model_path="",
        optimizer=ts.Optimizer.fire,
        convergence_fn=ConvergenceFn.FORCE,
        trajectory_reporter_dict=trajectory_reporter_dict,
        autobatcher_dict=autobatcher_dict,
        max_steps=500,
        steps_between_swaps=10,
        init_kwargs={"cell_filter": ts.CellFilter.unit},
        model_kwargs={"sigma": 3.405, "epsilon": 0.0104, "compute_stress": True},
    )

    job = maker.make([perturbed_structure] * n_systems)
    response_dict = run_locally(job, ensure_success=True, root_dir=tmp_path)
    result = list(response_dict.values())[-1][1].output

    # Validate result structure (TSTaskDoc)
    assert hasattr(result, "structures")
    assert hasattr(result, "calcs_reversed")
    assert hasattr(result, "time_elapsed")

    # Check structures list output
    assert isinstance(result.structures, list)
    assert len(result.structures) == n_systems
    assert isinstance(result.structures[0], Structure)

    # Check calculation details
    assert len(result.calcs_reversed) == 1
    calc = result.calcs_reversed[0]

    # Check model name
    assert calc.model == TSModelType.LENNARD_JONES
    assert calc.model_path is not None

    # Check optimizer
    assert calc.optimizer == ts.Optimizer.fire

    # Check trajectory reporter details
    assert calc.trajectory_reporter is not None
    assert calc.trajectory_reporter.state_frequency == 5
    assert hasattr(calc.trajectory_reporter, "prop_calculators")
    assert all(Path(f).is_file() for f in calc.trajectory_reporter.filenames)

    # Check autobatcher details
    assert calc.autobatcher is None

    # Check other parameters
    assert calc.max_steps == 500
    assert calc.steps_between_swaps == 10
    assert calc.init_kwargs["cell_filter"] == ts.CellFilter.unit

    # Check time elapsed
    assert result.time_elapsed > 0


def test_relax_job_mace(
    ar_structure: Structure, mace_model_path: str, tmp_path
) -> None:
    """Test TSOptimizeMaker with MACE model.

    Includes trajectory reporter and autobatcher.
    """
    # Perturb the structure to make optimization meaningful
    perturbed_structure = ar_structure.copy()
    perturbed_structure.translate_sites(
        list(range(len(perturbed_structure))), [0.01, 0.01, 0.01]
    )

    n_systems = 2
    trajectory_reporter_dict = {
        "filenames": [tmp_path / f"relax_{i}.h5md" for i in range(n_systems)],
        "state_frequency": 5,
        "prop_calculators": {1: ["potential_energy"]},
    }

    autobatcher_dict = {"memory_scales_with": "n_atoms", "max_memory_scaler": 260}

    maker = TSOptimizeMaker(
        model_type=TSModelType.MACE,
        model_path=mace_model_path,
        optimizer=ts.Optimizer.fire,
        convergence_fn=ConvergenceFn.FORCE,
        trajectory_reporter_dict=trajectory_reporter_dict,
        autobatcher_dict=autobatcher_dict,
        max_steps=500,
        steps_between_swaps=10,
        init_kwargs={"cell_filter": ts.CellFilter.unit},
    )

    job = maker.make([perturbed_structure] * n_systems)
    response_dict = run_locally(job, ensure_success=True, root_dir=tmp_path)
    result = list(response_dict.values())[-1][1].output

    # Validate result structure
    assert hasattr(result, "structures")
    assert len(result.structures) == n_systems
    assert len(result.calcs_reversed) == 1

    calc = result.calcs_reversed[0]
    assert calc.model == TSModelType.MACE
    assert calc.autobatcher is not None
    assert calc.autobatcher.memory_scales_with == "n_atoms"


def test_md_job_comprehensive(ar_structure: Structure, tmp_path) -> None:
    """Test TSIntegrateMaker with all kwargs.

    Includes trajectory reporter and autobatcher.
    """
    n_systems = 2
    trajectory_reporter_dict = {
        "filenames": [tmp_path / f"md_{i}.h5md" for i in range(n_systems)],
        "state_frequency": 2,
        "prop_calculators": {1: ["potential_energy", "kinetic_energy", "temperature"]},
    }

    # Create autobatcher
    autobatcher_dict = False

    maker = TSIntegrateMaker(
        model_type=TSModelType.LENNARD_JONES,
        model_path="",
        integrator=ts.Integrator.nvt_langevin,
        n_steps=20,
        temperature=300.0,
        timestep=0.001,
        trajectory_reporter_dict=trajectory_reporter_dict,
        autobatcher_dict=autobatcher_dict,
        model_kwargs={"sigma": 3.405, "epsilon": 0.0104, "compute_stress": True},
    )

    job = maker.make([ar_structure] * n_systems)
    response_dict = run_locally(job, ensure_success=True, root_dir=tmp_path)
    result = list(response_dict.values())[-1][1].output

    # Validate result structure (TSTaskDoc)
    assert hasattr(result, "structures")
    assert hasattr(result, "calcs_reversed")
    assert hasattr(result, "time_elapsed")

    # Check structures list output
    assert isinstance(result.structures, list)
    assert len(result.structures) == n_systems
    assert isinstance(result.structures[0], Structure)

    # Check calculation details
    assert len(result.calcs_reversed) == 1
    calc = result.calcs_reversed[0]

    # Check model name
    assert calc.model == TSModelType.LENNARD_JONES
    assert calc.model_path is not None

    # Check integrator
    assert calc.integrator == ts.Integrator.nvt_langevin

    # Check MD parameters
    assert calc.n_steps == 20
    assert calc.temperature == 300.0
    assert calc.timestep == 0.001

    # Check trajectory reporter details
    assert calc.trajectory_reporter is not None
    assert calc.trajectory_reporter.state_frequency == 2
    assert hasattr(calc.trajectory_reporter, "prop_calculators")
    assert all(Path(f).is_file() for f in calc.trajectory_reporter.filenames)

    # Check autobatcher details
    assert calc.autobatcher is None

    # Check time elapsed
    assert result.time_elapsed > 0


def test_static_job_comprehensive(ar_structure: Structure, tmp_path) -> None:
    """Test TSStaticMaker with all kwargs.

    Includes trajectory reporter and autobatcher.
    """
    n_systems = 2
    trajectory_reporter_dict = {
        "filenames": [tmp_path / f"static_{i}.h5md" for i in range(n_systems)],
        "state_frequency": 1,
        "prop_calculators": {1: ["potential_energy"]},
        "state_kwargs": {"save_forces": True},
    }

    # Create autobatcher
    autobatcher_dict = False

    maker = TSStaticMaker(
        model_type=TSModelType.LENNARD_JONES,
        model_path="",
        trajectory_reporter_dict=trajectory_reporter_dict,
        autobatcher_dict=autobatcher_dict,
        model_kwargs={"sigma": 3.405, "epsilon": 0.0104, "compute_stress": True},
    )

    job = maker.make([ar_structure] * n_systems)
    response_dict = run_locally(job, ensure_success=True, root_dir=tmp_path)
    result = list(response_dict.values())[-1][1].output

    # Validate result structure (TSTaskDoc)
    assert hasattr(result, "structures")
    assert hasattr(result, "calcs_reversed")
    assert hasattr(result, "time_elapsed")

    # Check structures list output
    assert isinstance(result.structures, list)
    assert len(result.structures) == n_systems
    assert isinstance(result.structures[0], Structure)

    # Check calculation details
    assert len(result.calcs_reversed) == 1
    calc = result.calcs_reversed[0]

    # Check model name
    assert calc.model == TSModelType.LENNARD_JONES
    assert calc.model_path is not None

    # Check trajectory reporter details
    assert calc.trajectory_reporter is not None
    assert calc.trajectory_reporter.state_frequency == 1
    assert hasattr(calc.trajectory_reporter, "prop_calculators")
    assert hasattr(calc.trajectory_reporter, "state_kwargs")
    assert calc.trajectory_reporter.state_kwargs["save_forces"] is True
    assert all(Path(f).is_file() for f in calc.trajectory_reporter.filenames)

    # Check autobatcher details
    assert calc.autobatcher is None

    # Check that all_properties is present
    assert hasattr(calc, "all_properties")
    assert isinstance(calc.all_properties, list)
    assert len(calc.all_properties) == n_systems

    # Check time elapsed
    assert result.time_elapsed > 0
