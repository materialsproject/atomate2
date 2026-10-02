"""Tests for the force field pheasy workflow.

pheasy and ALM are only installed in the numpy-limited forcefield CI job, so the
tests are skipped in the other forcefield jobs.
"""

import pytest

pytest.importorskip("pheasy")

from ase.build import bulk
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import atomate2.common.jobs.pheasy as pheasy_jobs
from atomate2.forcefields.flows.pheasy import PhononMaker


def _cu_structure() -> Structure:
    return AseAtomsAdaptor.get_structure(bulk("Cu", "fcc", a=3.61, cubic=True))


def test_get_supercell_size_kwargs(monkeypatch):
    received = {}
    transformation = pheasy_jobs.CubicSupercellTransformation

    def record_kwargs(**kwargs):
        received.update(kwargs)
        return transformation(**kwargs)

    monkeypatch.setattr(pheasy_jobs, "CubicSupercellTransformation", record_kwargs)

    # the maker passes get_supercell_size_kwargs on to the job
    maker = PhononMaker(get_supercell_size_kwargs={"angle_tolerance": 0.1})
    job = maker.get_supercell_matrix(_cu_structure())
    assert job.function_kwargs == {"angle_tolerance": 0.1}

    # the job passes them to CubicSupercellTransformation. The other default
    # is kept.
    job.function(*job.function_args, **job.function_kwargs)
    assert received["angle_tolerance"] == 0.1
    assert received["allow_orthorhombic"] is False
