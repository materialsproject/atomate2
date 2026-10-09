"""Tests for the force field CALPHAD workflow with ASE's EMT potential on Cu-Ni."""

import re
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
    # 10 A supercells give the same EMT entropies as 20 A to 1e-4 k_B/atom
    maker.phonon_maker.min_length = 10.0

    flow = maker.make(["Cu", "Ni"])
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    doc = responses[flow.output.uuid][1].output
    assert len(doc.calculations) == 6

    solids = [calc for calc in doc.calculations if calc.lattice != "LIQUID"]
    assert all(calc.is_force_converged for calc in solids)
    assert max(calc.relaxation_strain for calc in solids) < 0.1
    assert all(calc.imaginary_fraction == 0 for calc in solids)
    # EMT vibrational entropy in k_B/atom of the end members and the SQS
    entropy = [calc.vibrational_entropy / len(calc.structure) for calc in solids]
    assert entropy == pytest.approx([10.543, 9.748, 10.131], abs=1e-3)
    liquid = [calc for calc in doc.calculations if calc.lattice == "LIQUID"]
    # a crystal stays well below 1 A^2, the melt diffuses much further
    assert min(calc.mean_squared_displacement for calc in liquid) > 1
    assert all(calc.energy_standard_error > 0 for calc in liquid)
    # the 32-atom liquid at 2000 K is about 0.54 eV/atom above the relaxed solid
    solid = {calc.folder: calc.energy / len(calc.structure) for calc in solids}
    for calc in liquid:
        assert calc.energy / 32 - solid[calc.folder] == pytest.approx(0.54, abs=0.05)

    line = next(line for line in doc.tdb.splitlines() if "L(FCC_A1,CU,NI;0)" in line)
    # EMT mixing energy of the relaxed 32-atom SQS, 0.0212 eV/atom, and its
    # excess vibrational entropy, -0.015 k_B/atom
    enthalpy, entropy_term = re.fullmatch(
        r"([-+][\d.]+)([-+][\d.]+)\*T", line.split()[3]
    ).groups()
    assert float(enthalpy) == pytest.approx(8169.4, abs=1)
    assert float(entropy_term) == pytest.approx(0.503, abs=0.01)


def test_calphad_maker_needs_liquid_makers():
    with pytest.raises(ValueError, match="LIQUID needs"):
        CalphadMaker().make(["Cu", "Ni"])
