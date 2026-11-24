from __future__ import annotations

from types import SimpleNamespace

import pytest
from custodian.vasp.handlers import ErrorHandler
from pymatgen.io.vasp import Incar

from atomate2.vasp.run import DEFAULT_HANDLERS, run_vasp


def test_default_handlers():
    assert len(DEFAULT_HANDLERS) >= 8
    assert all(isinstance(handler, ErrorHandler) for handler in DEFAULT_HANDLERS)


@pytest.mark.parametrize(
    ("incar", "cmd", "gamma_cmd"),
    [
        ({}, ["vasp_std"], ["vasp_gam"]),
        ({"LSORBIT": True}, ["vasp_ncl"], None),
        ({"LNONCOLLINEAR": True}, ["vasp_ncl"], None),
    ],
)
def test_run_vasp_noncollinear(tmp_path, monkeypatch, incar, cmd, gamma_cmd):
    monkeypatch.chdir(tmp_path)
    Incar(incar).write_file("INCAR")
    jobs = []
    monkeypatch.setattr(
        "atomate2.vasp.run.VaspJob",
        lambda vasp_cmd, **kwargs: jobs.append((vasp_cmd, kwargs)),
    )
    monkeypatch.setattr(
        "atomate2.vasp.run.Custodian",
        lambda *_, **__: SimpleNamespace(run=lambda: None),
    )
    run_vasp(vasp_cmd="vasp_std", vasp_gamma_cmd="vasp_gam", vasp_ncl_cmd="vasp_ncl")
    assert jobs[0][0] == cmd
    assert jobs[0][1].get("gamma_vasp_cmd") == gamma_cmd
