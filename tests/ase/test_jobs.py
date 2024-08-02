"""Test the base ASE jobs."""
from __future__ import annotations
from jobflow import run_locally
from pymatgen.core import Structure, Lattice
import pytest

from atomate2.ase.jobs import LennardJonesRelaxMaker
from atomate2.ase.schemas import AseTaskDocument

def test_lennard_jones_relax_maker(lj_fcc_ne_pars, fcc_ne_structure):

    job = LennardJonesRelaxMaker(
        calculator_kwargs = lj_fcc_ne_pars,
        relax_kwargs = {
            "fmax": 0.001
        }
    ).make(fcc_ne_structure)

    resp = run_locally(job)
    output = resp[job.uuid][1].output

    assert output.structure.volume == pytest.approx(22.304245)
    assert output.output.energy == pytest.approx(-0.018494767)
    assert isinstance(output, AseTaskDocument)