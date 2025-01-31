from pathlib import Path

import numpy as np
import pytest
from emmet.core.openmm import OpenMMInterchange
from pymatgen.core import Structure

from atomate2.openmm.jobs.base import BaseOpenMMMaker
from atomate2.openmm.utils import (
    PymatgenTrajectoryReporter,
    download_opls_xml,
    increment_name,
    interchange_to_structure,
    structure_to_topology,
)


@pytest.mark.skip("Unreliable test, needs browser to run successfully.")
def test_download_xml(tmp_path: Path) -> None:
    pytest.importorskip("selenium")

    download_opls_xml("CCO", tmp_path / "CCO.xml")

    assert (tmp_path / "CCO.xml").exists()


def test_increment_file_name() -> None:
    test_cases = [
        ("report", "report2"),
        ("report123", "report124"),
        ("report.123", "report.124"),
        ("report-123", "report-124"),
        ("report-dcd", "report-dcd2"),
        ("report.123.dcd", "report.123.dcd2"),
    ]

    for file_name, expected_output in test_cases:
        result = increment_name(file_name)
        assert result == expected_output, (
            f"Failed for case: {file_name}. Expected: {expected_output}, Got: {result}"
        )


def test_trajectory_reporter(interchange: OpenMMInterchange, tmp_path: Path) -> None:
    """Test that the trajectory reporter correctly accumulates and formats data."""
    # Create simulation using BaseOpenMMMaker
    maker = BaseOpenMMMaker(
        temperature=300,
        friction_coefficient=1.0,
        step_size=0.002,
        platform_name="CPU",
    )
    simulation = maker._create_simulation(interchange)  # noqa: SLF001

    # Add trajectory reporter
    reporter = PymatgenTrajectoryReporter(
        file=tmp_path / "trajectory.json",
        reportInterval=1,
        enforcePeriodicBox=True,
    )
    simulation.reporters.append(reporter)

    # Run simulation for a few steps
    n_steps = 3
    simulation.step(n_steps)

    reporter.save()

    # Check trajectory was created
    assert hasattr(reporter, "trajectory")
    traj = reporter.trajectory

    # Check basic properties
    assert len(traj) == n_steps, (
        f"got {len(traj)=}, expected {n_steps=}"
    )  # should have n_steps frames
    assert traj.time_step is not None
    assert len(traj.species) == len(list(simulation.topology.atoms())), (
        f"got {len(traj.species)=}, expected {len(list(simulation.topology.atoms()))=}"
    )

    # Check frame properties
    assert len(traj.frame_properties) == n_steps
    for frame in traj.frame_properties:
        assert isinstance(frame["kinetic_energy"], float)
        assert isinstance(frame["potential_energy"], float)
        assert isinstance(frame["total_energy"], float)

    # Check site properties
    assert len(traj.site_properties) == n_steps
    for frame in traj.site_properties:
        assert "velocities" in frame
        # Check velocities are stored as tuples
        assert isinstance(frame["velocities"][0], tuple)
        assert len(frame["velocities"][0]) == 3

    # check that file was written
    assert (tmp_path / "trajectory.json").exists()


@pytest.mark.openmm_slow
def test_structure_to_topology(random_structure: Structure) -> None:
    topology = structure_to_topology(random_structure)
    assert topology is not None, "Topology should not be None."
    num_atoms_in_topology = sum(1 for _ in topology.atoms())
    assert num_atoms_in_topology == len(random_structure), (
        "Number of atoms in topology should match structure."
    )


@pytest.mark.openmm_slow
def test_interchange_to_structure(interchange: OpenMMInterchange) -> None:
    structure = interchange_to_structure(interchange)
    assert len(structure) == 1170
    assert 4 < np.max(structure.cart_coords) < 16
