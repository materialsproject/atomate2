"""Test the base ASE jobs."""

from __future__ import annotations

import os

import pytest
from jobflow import run_locally
from pymatgen.core import Molecule

from atomate2.ase.jobs import (
    GFNxTBRelaxMaker,
    GFNxTBStaticMaker,
    LennardJonesRelaxMaker,
    LennardJonesStaticMaker,
)
from atomate2.ase.schemas import AseMoleculeTaskDoc, AseStructureTaskDoc

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
    assert isinstance(output, AseStructureTaskDoc)
    assert fcc_ne_structure.matches(
        output.structure
    ), f"{output.structure} != {fcc_ne_structure}"


def test_lennard_jones_static_maker(lj_fcc_ne_pars, fcc_ne_structure):
    job = LennardJonesStaticMaker(calculator_kwargs=lj_fcc_ne_pars).make(
        fcc_ne_structure
    )
    response = run_locally(job)
    output = response[job.uuid][1].output

    assert output.output.energy == pytest.approx(-0.0179726955438795)
    assert output.structure.volume == pytest.approx(24.334)
    assert isinstance(output, AseStructureTaskDoc)

    # Structure.__eq__ checks properties which contains 'energy', 'forces', 'stress'
    # so need to reset properties to ensure equality
    output.structure.properties = fcc_ne_structure.properties
    assert (
        output.structure == fcc_ne_structure
    ), f"{output.structure} != {fcc_ne_structure}"


@pytest.mark.skipif(condition=TBLite is None, reason="TBLite must be installed.")
def test_gfn_xtb_relax_maker(h2o_3uud_trimer):
    os.environ["OMP_NUM_THREADS"] = "1"
    job = GFNxTBRelaxMaker(
        calculator_kwargs={
            "method": "GFN2-xTB",
        },
        relax_kwargs={"fmax": 0.01},
    ).make(h2o_3uud_trimer)

    response = run_locally(job)
    output = response[job.uuid][1].output

    expected_relaxed_molecule = Molecule.from_str(
        """9
H6 O3
O -1.405082 -0.755173 -0.152341
H -0.494514 -1.024689 -0.365082
H -1.748321 -1.406728 0.460615
O 1.379213 -0.766691 -0.269151
H 1.162911 0.173598 -0.132786
H 1.968668 -1.011288 0.445777
O 0.020787 1.592087 0.237523
H -0.701636 0.940936 0.196638
H -0.181997 2.257930 -0.421222""",
        fmt="xyz",
    )
    for isite, site in enumerate(expected_relaxed_molecule):
        assert output.molecule[isite].species == site.species
        assert all(
            output.molecule[isite].coords[i]
            == pytest.approx(site.coords[i], abs=1.0e-5)
            for i in range(3)
        )
    assert output.output.energy_per_atom == pytest.approx(-46.06280925889291)
    assert output.energy_downhill
    assert output.is_force_converged
    assert isinstance(output, AseMoleculeTaskDoc)


@pytest.mark.skipif(condition=TBLite is None, reason="TBLite must be installed.")
def test_gfn_xtb_static_maker(h2o_3uud_trimer):
    os.environ["OMP_NUM_THREADS"] = "1"
    job = GFNxTBStaticMaker(
        calculator_kwargs={"method": "GFN2-xTB"},
    ).make(h2o_3uud_trimer)

    response = run_locally(job)
    output = response[job.uuid][1].output

    assert output.output.energy_per_atom == pytest.approx(-46.05920227158222)
    assert isinstance(output, AseMoleculeTaskDoc)

    # Molecule.__eq__ checks properties which contains 'energy', 'forces', 'stress'
    # so need to reset properties to ensure equality
    output.molecule.properties = h2o_3uud_trimer.properties
    assert output.molecule == h2o_3uud_trimer
