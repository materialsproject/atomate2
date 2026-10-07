"""Tests for the force field CALPHAD workflow with ASE's EMT potential on Cu-Ni."""

import shutil

import pytest
from jobflow import run_locally

from atomate2.forcefields.flows.calphad import CalphadMaker

EMT = {"@module": "ase.calculators.emt", "@callable": "EMT"}


@pytest.mark.skipif(shutil.which("sqs2tdb") is None, reason="ATAT is not installed")
def test_calphad_maker_emt(clean_dir):
    lattices = ["FCC_A1", "LIQUID"]
    maker = CalphadMaker.from_force_field_name(
        EMT,
        melt_temperature=2500,
        liquid_temperature=2000,
        lattices=lattices,
        level=1,
        terms={lattice: ["1,0", "2,0"] for lattice in lattices},
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
    assert all(calc.energy_standard_error > 0 for calc in liquid)
    # the 32-atom liquid at 2000 K is about 0.54 eV/atom above the relaxed solid
    solid = {calc.folder: calc.energy / len(calc.structure) for calc in solids}
    for calc in liquid:
        assert calc.energy / 32 - solid[calc.folder] == pytest.approx(0.54, abs=0.05)

    line = next(line for line in doc.tdb.splitlines() if "L(FCC_A1,CU,NI;0)" in line)
    # EMT mixing energy of the relaxed 32-atom SQS, 0.0212 eV/atom
    assert float(line.split()[3]) == pytest.approx(8168.4, abs=1)


def test_calphad_maker_needs_liquid_makers():
    with pytest.raises(ValueError, match="LIQUID needs"):
        CalphadMaker().make(["Cu", "Ni"])
