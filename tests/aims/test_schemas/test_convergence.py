"""Test that FHI-aims convergence is reflected in the task document state."""

import os

import pytest
from jobflow import run_locally
from pymatgen.io.aims.outputs import AimsOutput
from pymatgen.io.aims.sets.core import StaticSetGenerator

from atomate2.aims.jobs.core import StaticMaker
from atomate2.aims.run import should_stop_children
from atomate2.aims.schemas.calculation import TaskState

cwd = os.getcwd()


def _run_static(si, tmp_path, mock_aims, species_dir, ref_dir):
    """Run a mocked static calculation using the outputs in the given reference dir."""
    mock_aims({"base": ref_dir}, {})

    parameters = {
        "k_grid": [2, 2, 2],
        "species_dir": (species_dir / "light").as_posix(),
    }
    maker = StaticMaker(input_set_generator=StaticSetGenerator(user_params=parameters))
    maker.name = "base"
    job = maker.make(si)

    os.chdir(tmp_path)
    try:
        responses = run_locally(job, create_folders=True, ensure_success=True)
    finally:
        os.chdir(cwd)
    return responses[job.uuid][1]


def test_converged_calculation(si, tmp_path, mock_aims, species_dir):
    response = _run_static(si, tmp_path, mock_aims, species_dir, "static-si")
    doc = response.output
    calc = doc.calcs_reversed[-1]

    assert calc.completed
    assert calc.converged
    assert calc.has_aims_completed == TaskState.SUCCESS
    assert doc.completed
    assert doc.converged
    assert doc.state == TaskState.SUCCESS
    assert not response.stop_children
    assert not should_stop_children(doc)


def test_unconverged_calculation(si, tmp_path, mock_aims, species_dir):
    # same calculation as static-si, without the "SCF cycle converged" line
    response = _run_static(
        si, tmp_path, mock_aims, species_dir, "static-si-unconverged"
    )
    doc = response.output
    calc = doc.calcs_reversed[-1]

    # the run finished and produced results, but did not converge
    assert calc.completed
    assert not calc.converged
    assert calc.has_aims_completed == TaskState.FAILED
    assert doc.completed
    assert not doc.converged
    assert doc.state == TaskState.FAILED
    assert response.stop_children

    assert should_stop_children(doc)
    assert should_stop_children(doc, handle_unsuccessful=True)
    assert not should_stop_children(doc, handle_unsuccessful=False)
    with pytest.raises(RuntimeError, match="not converged"):
        should_stop_children(doc, handle_unsuccessful="error")


def test_unknown_convergence(si, tmp_path, mock_aims, species_dir, monkeypatch):
    # convergence unknown (e.g. output serialized by an older pymatgen version):
    # the state only depends on whether the calculation completed
    monkeypatch.setattr(
        AimsOutput, "converged", property(lambda _: None), raising=False
    )
    response = _run_static(si, tmp_path, mock_aims, species_dir, "static-si")
    doc = response.output
    calc = doc.calcs_reversed[-1]

    assert calc.completed
    assert calc.converged is None
    assert calc.has_aims_completed == TaskState.SUCCESS
    assert doc.converged is None
    assert doc.state == TaskState.SUCCESS
    assert not response.stop_children
