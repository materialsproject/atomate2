"""Tests for TorchSim phonons workflow."""
# ruff: noqa: E402

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("ase")
ts = pytest.importorskip("torch_sim")

from jobflow import Flow, run_locally

from atomate2.common.jobs.phonons import (
    generate_phonon_displacements,
    get_supercell_size,
)
from atomate2.torchsim.core import TorchSimOptimizeMaker, TorchSimStaticMaker
from atomate2.torchsim.flows.phonons import PhononMaker
from atomate2.torchsim.schema import TorchSimModelType

if TYPE_CHECKING:
    from pymatgen.core import Structure


def test_torchsim_phonon_displacements(si_diamond: Structure, tmp_path) -> None:
    """Test TorchSimStaticMaker can compute forces on phonon displaced structures.

    This test validates that TorchSim's static maker produces output compatible
    with the phonon workflow interface. It tests:
    1. Phonon displacement generation using standard atomate2 machinery
    2. Batch force calculation using TorchSim
    3. Output schema compatibility (task_doc.output.all_forces and .forces)
    """
    # Step 1: Get supercell size (using small supercell for fast testing)
    supercell_job = get_supercell_size(
        si_diamond, min_length=8, max_length=12, prefer_90_degrees=True
    )
    responses = run_locally(supercell_job, create_folders=True, ensure_success=True)
    supercell_matrix = responses[supercell_job.uuid][1].output

    # Step 2: Generate phonon displacements
    displacement_job = generate_phonon_displacements(
        structure=si_diamond,
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


def test_torchsim_output_schema_compatibility(si_diamond: Structure, tmp_path) -> None:
    """Test that TorchSimTaskDoc output schema matches phonon workflow expectations.

    The phonon workflow (run_phonon_displacements) accesses:
    - task_doc.output.all_forces for socket/batch mode
    - task_doc.output.forces for non-socket/single mode

    This test verifies the schema structure is correct.
    """
    maker = TorchSimStaticMaker(
        model_type=TorchSimModelType.LENNARD_JONES,
        model_path="",
        model_kwargs={"sigma": 2.0, "epsilon": 0.01, "compute_stress": True},
    )

    # Test with multiple structures (batch mode)
    job = maker.make([si_diamond, si_diamond])
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


@pytest.mark.parametrize("socket", [True, False])
def test_torchsim_phonon_maker_integration(
    si_diamond: Structure, tmp_path, socket: bool
) -> None:
    """Test that TorchSim makers can be used within PhononMaker.

    This test validates that TorchSimOptimizeMaker and TorchSimStaticMaker
    can be used as bulk_relax_maker and static_energy_maker within PhononMaker,
    ensuring proper schema compatibility for phonon workflow integration.
    """
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

    phonon_maker = PhononMaker(
        bulk_relax_maker=relax_maker,
        static_energy_maker=static_maker,
        phonon_displacement_maker=static_maker,
        use_symmetrized_structure="primitive",  # required for non-seekpath kpath
        create_thermal_displacements=False,
        store_force_constants=False,
        kpath_scheme="setyawan_curtarolo",  # avoid seekpath dependency
        socket=socket,
    )

    # Create the phonon flow
    flow = phonon_maker.make(si_diamond)

    # Verify flow is created successfully
    assert isinstance(flow, Flow)
    assert len(flow) >= 5  # At minimum: conv, relax, supercell, static, displacements

    # Check that the TorchSim jobs are present in the flow
    job_names = [j.name for j in flow]
    assert "torchsim optimize" in job_names, f"Expected relax job, got {job_names}"
    assert "torchsim static" in job_names, f"Expected static job, got {job_names}"

    # Run the flow locally to verify end-to-end execution
    run_locally(flow, create_folders=True, ensure_success=True, root_dir=tmp_path)
