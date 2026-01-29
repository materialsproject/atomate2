"""Tests for TorchSim core makers."""
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path

import pytest

ts = pytest.importorskip("torch_sim")

from ase.build import bulk
from jobflow import run_locally
from mace.calculators.foundations_models import download_mace_mp_checkpoint
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.common.jobs.phonons import (
    generate_phonon_displacements,
    get_supercell_size,
)
from atomate2.torchsim.core import (
    TorchSimIntegrateMaker,
    TorchSimOptimizeMaker,
    TorchSimStaticMaker,
)
from atomate2.torchsim.schema import ConvergenceFn, TorchSimModelType


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

    maker = TorchSimOptimizeMaker(
        model_type=TorchSimModelType.LENNARD_JONES,
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
    assert calc.model == TorchSimModelType.LENNARD_JONES
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

    # Check calculation output (energy, forces, stress)
    assert calc.output is not None
    assert calc.output.energies is not None
    assert len(calc.output.energies) == n_systems
    assert all(isinstance(e, float) for e in calc.output.energies)
    assert calc.output.all_forces is not None
    assert len(calc.output.all_forces) == n_systems
    assert calc.output.stress is not None
    assert len(calc.output.stress) == n_systems

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

    maker = TorchSimOptimizeMaker(
        model_type=TorchSimModelType.MACE,
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
    assert calc.model == TorchSimModelType.MACE
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

    maker = TorchSimIntegrateMaker(
        model_type=TorchSimModelType.LENNARD_JONES,
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
    assert calc.model == TorchSimModelType.LENNARD_JONES
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

    # Check calculation output (energy, forces, stress)
    assert calc.output is not None
    assert calc.output.energies is not None
    assert len(calc.output.energies) == n_systems
    assert all(isinstance(e, float) for e in calc.output.energies)
    assert calc.output.all_forces is not None
    assert len(calc.output.all_forces) == n_systems
    assert calc.output.stress is not None
    assert len(calc.output.stress) == n_systems

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
        "prop_calculators": {1: ["potential_energy", "forces", "stress"]},
    }

    # Create autobatcher
    autobatcher_dict = False

    maker = TorchSimStaticMaker(
        model_type=TorchSimModelType.LENNARD_JONES,
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
    assert calc.model == TorchSimModelType.LENNARD_JONES
    assert calc.model_path is not None

    # Check trajectory reporter details
    assert calc.trajectory_reporter is not None
    assert calc.trajectory_reporter.state_frequency == 1
    assert hasattr(calc.trajectory_reporter, "prop_calculators")
    assert all(Path(f).is_file() for f in calc.trajectory_reporter.filenames)

    # Check autobatcher details
    assert calc.autobatcher is None

    # Check that all_properties is present
    assert hasattr(calc, "all_properties")
    assert isinstance(calc.all_properties, list)
    assert len(calc.all_properties) == n_systems

    # Check calculation output (energy, forces, stress)
    assert calc.output is not None
    assert calc.output.energies is not None
    assert len(calc.output.energies) == n_systems
    assert all(isinstance(e, float) for e in calc.output.energies)
    assert calc.output.all_forces is not None
    assert len(calc.output.all_forces) == n_systems
    assert calc.output.stress is not None
    assert len(calc.output.stress) == n_systems

    # Check time elapsed
    assert result.time_elapsed > 0


@pytest.fixture
def si_structure():
    """Create a silicon structure for testing."""
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    return AseAtomsAdaptor.get_structure(atoms)


def test_torchsim_phonon_displacements(si_structure: Structure, tmp_path) -> None:
    """Test TorchSimStaticMaker can compute forces on phonon displaced structures.

    This test validates that TorchSim's static maker produces output compatible
    with the phonon workflow interface. It tests:
    1. Phonon displacement generation using standard atomate2 machinery
    2. Batch force calculation using TorchSim
    3. Output schema compatibility (task_doc.output.all_forces and .forces)
    """
    # Step 1: Get supercell size (using small supercell for fast testing)
    supercell_job = get_supercell_size(
        si_structure, min_length=8, max_length=12, prefer_90_degrees=True
    )
    responses = run_locally(supercell_job, create_folders=True, ensure_success=True)
    supercell_matrix = responses[supercell_job.uuid][1].output

    # Step 2: Generate phonon displacements
    displacement_job = generate_phonon_displacements(
        structure=si_structure,
        supercell_matrix=supercell_matrix,
        displacement=0.01,
        sym_reduce=True,
        symprec=1e-4,
        use_symmetrized_structure=None,
        kpath_scheme="seekpath",
        code="torchsim",
    )
    responses = run_locally(displacement_job, create_folders=True, ensure_success=True)
    displaced_structures = responses[displacement_job.uuid][1].output

    # Verify we have displacements to test
    assert len(displaced_structures) > 0, "No displaced structures generated"

    # Step 3: Compute forces using TorchSim (batched calculation)
    # Using Lennard-Jones for testing (works without external model files)
    maker = TorchSimStaticMaker(
        model_type=TorchSimModelType.LENNARD_JONES,
        model_path="",
        model_kwargs={"sigma": 2.0, "epsilon": 0.01, "compute_stress": True},
    )

    # Run static calculation on all displaced structures at once
    # This demonstrates TorchSim's native batch processing capability
    job = maker.make(displaced_structures)
    response_dict = run_locally(job, ensure_success=True, root_dir=tmp_path)
    task_doc = list(response_dict.values())[-1][1].output

    # Step 4: Validate phonon-compatible output interface
    # The phonon workflow accesses task_doc.output.all_forces (batch mode)
    # and task_doc.output.forces (single structure mode)
    assert hasattr(task_doc, "output"), "TorchSimTaskDoc must have output property"

    output = task_doc.output
    assert output.all_forces is not None, "all_forces should be populated"
    assert len(output.all_forces) == len(displaced_structures)

    # Verify force dimensions match atom counts
    for i, (forces, struct) in enumerate(
        zip(output.all_forces, displaced_structures, strict=True)
    ):
        assert len(forces) == len(struct), (
            f"Force count mismatch for structure {i}: "
            f"got {len(forces)}, expected {len(struct)}"
        )
        # Each force should be a 3D vector
        for atom_force in forces:
            assert len(atom_force) == 3, f"Force should be 3D vector, got {atom_force}"

    # Test single-structure access via .forces property
    assert output.forces is not None, "forces property should return first structure"
    assert len(output.forces) == len(displaced_structures[0])

    # Verify energies are computed
    assert output.energies is not None
    assert len(output.energies) == len(displaced_structures)


def test_torchsim_output_schema_compatibility(
    ar_structure: Structure, tmp_path
) -> None:
    """Test that TorchSimTaskDoc output schema matches phonon workflow expectations.

    The phonon workflow (run_phonon_displacements) accesses:
    - task_doc.output.all_forces for socket/batch mode
    - task_doc.output.forces for non-socket/single mode

    This test verifies the schema structure is correct.
    """
    maker = TorchSimStaticMaker(
        model_type=TorchSimModelType.LENNARD_JONES,
        model_path="",
        model_kwargs={"sigma": 3.405, "epsilon": 0.0104, "compute_stress": True},
    )

    # Test with multiple structures (batch mode)
    job = maker.make([ar_structure, ar_structure])
    response_dict = run_locally(job, ensure_success=True, root_dir=tmp_path)
    task_doc = list(response_dict.values())[-1][1].output

    # Verify the output access pattern matches phonon expectations
    # Phonon code does: phonon_job.output.output.all_forces
    # With jobflow output_schema, this becomes: task_doc.output.all_forces
    assert task_doc.output.all_forces is not None
    assert len(task_doc.output.all_forces) == 2

    # Verify .forces returns first structure's forces
    assert task_doc.output.forces is not None
    assert task_doc.output.forces == task_doc.output.all_forces[0]

    # Verify stress tensor format
    assert task_doc.output.stress is not None
    assert len(task_doc.output.stress) == 2
    # Each stress should be a 3x3 matrix
    for stress in task_doc.output.stress:
        assert len(stress) == 3
        for row in stress:
            assert len(row) == 3


def test_torchsim_phonon_maker_integration(si_structure: Structure, tmp_path) -> None:
    """Test that TorchSim makers can be used within PhononMaker.

    This test validates that TorchSimOptimizeMaker and TorchSimStaticMaker
    can be used as bulk_relax_maker and static_energy_maker within PhononMaker,
    ensuring proper schema compatibility for phonon workflow integration.
    """
    from dataclasses import dataclass

    from jobflow import Flow

    from atomate2.common.flows.phonons import BasePhononMaker

    # Create TorchSim makers for phonon workflow
    relax_maker = TorchSimOptimizeMaker(
        model_type=TorchSimModelType.LENNARD_JONES,
        model_path="",
        optimizer=ts.Optimizer.fire,
        model_kwargs={"sigma": 2.0, "epsilon": 0.01, "compute_stress": True},
        max_steps=100,
        init_kwargs={"cell_filter": ts.CellFilter.unit},
    )

    static_maker = TorchSimStaticMaker(
        model_type=TorchSimModelType.LENNARD_JONES,
        model_path="",
        model_kwargs={"sigma": 2.0, "epsilon": 0.01, "compute_stress": True},
    )

    # Create a minimal PhononMaker subclass for testing
    @dataclass
    class TorchSimPhononMaker(BasePhononMaker):
        """Test PhononMaker using TorchSim makers."""

        name: str = "torchsim phonon"
        bulk_relax_maker: TorchSimOptimizeMaker | None = None
        static_energy_maker: TorchSimStaticMaker | None = None
        phonon_displacement_maker: TorchSimStaticMaker | None = None
        code: str = "torchsim"

        @property
        def prev_calc_dir_argname(self) -> None:
            """TorchSim doesn't use prev_calc_dir."""
            return None

    phonon_maker = TorchSimPhononMaker(
        bulk_relax_maker=relax_maker,
        static_energy_maker=static_maker,
        phonon_displacement_maker=static_maker,
        use_symmetrized_structure="primitive",  # required for non-seekpath kpath
        create_thermal_displacements=False,
        store_force_constants=False,
        kpath_scheme="setyawan_curtarolo",  # avoid seekpath dependency
    )

    # Create the phonon flow
    flow = phonon_maker.make(si_structure)

    # Verify flow is created successfully
    assert isinstance(flow, Flow)
    assert len(flow) >= 5  # At minimum: conv, relax, supercell, static, displacements

    # Check that the TorchSim jobs are present in the flow
    job_names = [j.name for j in flow]
    assert "torchsim optimize" in job_names, f"Expected relax job, got {job_names}"
    assert "torchsim static" in job_names, f"Expected static job, got {job_names}"

    # Run the flow locally to verify end-to-end execution
    run_locally(flow, create_folders=True, ensure_success=True, root_dir=tmp_path)
