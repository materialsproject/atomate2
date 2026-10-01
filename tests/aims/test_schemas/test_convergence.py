"""Test that FHI-aims convergence is reflected in the task document state."""

import gzip
import shutil
from pathlib import Path

import pytest
from pymatgen.io.aims.outputs import AimsOutput

from atomate2.aims.run import should_stop_children
from atomate2.aims.schemas.calculation import TaskState
from atomate2.aims.schemas.task import AimsTaskDoc

REF_DIR = Path(__file__).resolve().parents[2] / "test_data/aims/static-si"


@pytest.fixture
def aims_dir(tmp_path):
    """Copy the static-si reference calculation, uncompressed, to a temp dir."""
    for name in ("aims.out", "geometry.in"):
        with gzip.open(REF_DIR / f"{name}.gz", "rb") as src:
            with open(tmp_path / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    shutil.copy(REF_DIR / "parameters.json", tmp_path / "parameters.json")
    return tmp_path


def _set_converged(monkeypatch, value: bool) -> None:
    monkeypatch.setattr(
        AimsOutput, "converged", property(lambda _: value), raising=False
    )


def test_converged_calculation(aims_dir, monkeypatch):
    _set_converged(monkeypatch, True)
    doc = AimsTaskDoc.from_directory(aims_dir)
    calc = doc.calcs_reversed[-1]

    assert calc.completed
    assert calc.converged
    assert calc.has_aims_completed == TaskState.SUCCESS
    assert doc.completed
    assert doc.converged
    assert doc.state == TaskState.SUCCESS
    assert not should_stop_children(doc)


def test_unconverged_calculation(aims_dir, monkeypatch):
    _set_converged(monkeypatch, False)
    doc = AimsTaskDoc.from_directory(aims_dir)
    calc = doc.calcs_reversed[-1]

    # the run finished and produced results, but did not converge
    assert calc.completed
    assert not calc.converged
    assert calc.has_aims_completed == TaskState.FAILED
    assert doc.completed
    assert not doc.converged
    assert doc.state == TaskState.FAILED

    assert should_stop_children(doc)
    assert should_stop_children(doc, handle_unsuccessful=True)
    assert not should_stop_children(doc, handle_unsuccessful=False)
    with pytest.raises(RuntimeError, match="not converged"):
        should_stop_children(doc, handle_unsuccessful="error")
