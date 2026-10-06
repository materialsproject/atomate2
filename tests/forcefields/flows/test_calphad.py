"""Tests for the force field CALPHAD workflow with ASE's EMT potential on Cu-Ni."""

import shutil

import pytest
from jobflow import run_locally

from atomate2.forcefields.flows.calphad import CalphadMaker

pytestmark = pytest.mark.skipif(
    shutil.which("sqs2tdb") is None, reason="ATAT sqs2tdb is not installed"
)

EMT = {"@module": "ase.calculators.emt", "@callable": "EMT"}


def test_calphad_maker_emt(clean_dir):
    lattices = ["FCC_A1", "LIQUID"]
    maker = CalphadMaker.from_force_field_name(
        EMT,
        melt_temperature=2500,
        liquid_temperature=2000,
        lattices=lattices,
        level=1,
        n_equilibration_frames=50,
        liquid_supercell=1,
    )
    # short runs: 2 ps melt, 3 ps liquid MD with the first 1 ps left out
    maker.liquid_melt_maker.n_steps = 1000
    maker.liquid_md_maker.n_steps = 1500
    for md_maker in (maker.liquid_melt_maker, maker.liquid_md_maker):
        md_maker.mb_velocity_seed = 1234

    flow = maker.make(["Cu", "Ni"])
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    doc = responses[flow.output.uuid][1].output
    assert len(doc.calculations) == 6

    solids = [calc for calc in doc.calculations if calc.lattice != "LIQUID"]
    assert all(calc.is_force_converged for calc in solids)
    assert max(calc.relaxation_strain for calc in solids) < 0.1
    liquid = [calc for calc in doc.calculations if calc.lattice == "LIQUID"]
    # a crystal stays well below 1 A^2, the melt diffuses much further
    assert min(calc.mean_squared_displacement for calc in liquid) > 1

    l0 = {
        lattice: float(
            next(
                line for line in doc.tdb.splitlines() if f"L({lattice},CU,NI;0)" in line
            ).split()[3]
        )
        for lattice in lattices
    }
    # EMT mixing energy of the relaxed 32-atom SQS, 0.0211 eV/atom
    assert l0["FCC_A1"] == pytest.approx(8168.4, abs=1)
    # the MD trajectories differ between platforms, so the liquid value is not pinned
    assert "LIQUID" in l0


def test_calphad_maker_needs_liquid_makers():
    with pytest.raises(ValueError, match="LIQUID needs"):
        CalphadMaker().make(["Cu", "Ni"])
