import pytest
from ase.io.espresso import read_fortran_namelist

from atomate2 import SETTINGS
from atomate2.espresso.sets import (
    EspressoDielectricInputGenerator,
    EspressoInputGenerator,
)


@pytest.fixture
def pseudo_dir(monkeypatch, espresso_test_dir):
    monkeypatch.setattr(
        SETTINGS, "ESPRESSO_PSEUDO_DIR", str(espresso_test_dir / "pseudo")
    )


def test_input_generator(pseudo_dir, si_primitive):
    generator = EspressoInputGenerator(
        pseudo_family="test_family",
        kspacing=0.5,
        user_input_data={"system": {"degauss": 0.02}, "ions": {"ion_dynamics": "bfgs"}},
    )
    input_set = generator.get_input_set(si_primitive)
    namelists, cards = read_fortran_namelist(input_set["pw.in"].splitlines())

    assert namelists["control"]["calculation"] == "scf"
    assert namelists["control"]["pseudo_dir"] == SETTINGS.ESPRESSO_PSEUDO_DIR
    assert namelists["control"]["outdir"] == "./"
    assert namelists["control"]["disk_io"] == "nowf"
    assert namelists["system"]["ecutwfc"] == 20
    assert namelists["system"]["ecutrho"] == 80
    assert namelists["system"]["degauss"] == 0.02
    assert namelists["system"]["smearing"] == "cold"
    assert namelists["ions"]["ion_dynamics"] == "bfgs"
    assert namelists["electrons"]["conv_thr"] == pytest.approx(2e-10)
    # each reciprocal lattice vector is 1.99 1/Å long, so 4 points at 0.5 1/Å
    assert cards[cards.index("K_POINTS automatic") + 1].split() == ["4"] * 3 + ["0"] * 3
    assert "Si 28.085 Si.pbe-rrkj.UPF" in cards

    supercell = si_primitive * (2, 1, 1)
    namelists, _ = read_fortran_namelist(
        generator.get_input_set(supercell)["pw.in"].splitlines()
    )
    assert namelists["electrons"]["conv_thr"] == pytest.approx(4e-10)


def test_relax_settings(pseudo_dir, si_primitive):
    generator = EspressoInputGenerator(
        calculation="vc-relax",
        pseudo_family="test_family",
        user_input_data={"control": {"nstep": 100}},
    )
    input_data = generator.get_input_data(si_primitive)

    assert input_data["control"]["forc_conv_thr"] == 1e-4
    assert input_data["control"]["etot_conv_thr"] == 1e-6
    assert input_data["control"]["nstep"] == 100
    assert input_data["cell"]["press_conv_thr"] == 0.1


def test_input_generator_without_pseudo_dir(monkeypatch, si_primitive):
    monkeypatch.setattr(SETTINGS, "ESPRESSO_PSEUDO_DIR", None)
    with pytest.raises(ValueError, match="ESPRESSO_PSEUDO_DIR"):
        EspressoInputGenerator().get_input_set(si_primitive)


def test_dielectric_input_generator(pseudo_dir, si_primitive):
    structure = si_primitive.copy(site_properties={"magmom": [0.5, 0.5]})
    generator = EspressoDielectricInputGenerator(
        pseudo_family="test_family",
        user_input_data={"system": {"occupations": "smearing", "ecutwfc": 25}},
    )
    input_set = generator.get_input_set(structure)
    namelists, _ = read_fortran_namelist(input_set["pw.in"].splitlines())
    ph_namelists, ph_cards = read_fortran_namelist(input_set["ph.in"].splitlines())

    assert namelists["control"]["disk_io"] == "low"
    assert namelists["system"]["occupations"] == "fixed"
    assert namelists["system"]["ecutwfc"] == 25
    assert "degauss" not in namelists["system"]
    assert "nspin" not in namelists["system"]
    assert ph_namelists["inputph"] == {
        "outdir": "./",
        "tr2_ph": 1e-16,
        "epsil": True,
        "trans": False,
    }
    assert ph_cards[-1].split() == ["0.00000000"] * 3
