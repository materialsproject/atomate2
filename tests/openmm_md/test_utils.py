from pathlib import Path

import pytest
from emmet.core.openmm import OpenMMInterchange

from atomate2.openmm.jobs.base import BaseOpenMMMaker
from atomate2.openmm.utils import (
    PymatgenTrajectoryReporter,
    download_opls_xml,
    increment_name,
)


@pytest.mark.skip("annoying test")
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
