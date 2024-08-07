"""Test the base ASE jobs."""

from __future__ import annotations

import pytest
from jobflow import run_locally

from atomate2.ase.jobs import (
    GFNxTBRelaxMaker,
    GFNxTBStaticMaker,
    LennardJonesRelaxMaker,
    LennardJonesStaticMaker,
)
from atomate2.ase.schemas import AseTaskDocument

try:
    from tblite.ase import TBLite
except ImportError:
    TBLite = None


def test_lennard_jones_relax_maker(lj_fcc_ne_pars, fcc_ne_structure):
    job = LennardJonesRelaxMaker(
        calculator_kwargs=lj_fcc_ne_pars, relax_kwargs={"fmax": 0.001}
    ).make(fcc_ne_structure)

    response = run_locally(job)
    output = response[job.uuid][1].output

    assert output.structure.volume == pytest.approx(22.304245)
    assert output.output.energy == pytest.approx(-0.018494767)
    assert isinstance(output, AseTaskDocument)


def test_lennard_jones_static_maker(lj_fcc_ne_pars, fcc_ne_structure):
    job = LennardJonesStaticMaker(calculator_kwargs=lj_fcc_ne_pars).make(
        fcc_ne_structure
    )
    response = run_locally(job)
    output = response[job.uuid][1].output

    assert output.output.energy == pytest.approx(-0.0179726955438795)
    assert output.structure.volume == pytest.approx(24.334)
    assert isinstance(output, AseTaskDocument)
    assert output.structure == fcc_ne_structure


@pytest.mark.skipif(condition=TBLite is None, reason="TBLite must be installed.")
def test_gfn_xtb_relax_maker(si_structure):
    job = GFNxTBRelaxMaker(
        calculator_kwargs={
            "method": "GFN1-xTB",
        },
        relax_kwargs={"fmax": 0.01},
    ).make(si_structure)

    response = run_locally(job)
    output = response[job.uuid][1].output

    assert output.structure.volume == pytest.approx(46.36854064419928)
    assert output.output.energy == pytest.approx(-87.63153322348951)
    assert output.energy_downhill
    assert output.is_force_converged
    assert isinstance(output, AseTaskDocument)


@pytest.mark.skipif(condition=TBLite is None, reason="TBLite must be installed.")
def test_gfn_xtb_static_maker(si_structure):
    job = GFNxTBStaticMaker(
        calculator_kwargs={
            "method": "GFN2-xTB",
        },
    ).make(si_structure)

    response = run_locally(job)
    output = response[job.uuid][1].output

    assert output.structure.volume == pytest.approx(40.88829274510334)
    assert output.output.energy == pytest.approx(-85.12729944654562)
    assert isinstance(output, AseTaskDocument)
    assert output.structure == si_structure
