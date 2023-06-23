import pytest
from pytest import approx

from atomate2.vasp.flows.mp import MPMetaGGARelax
from atomate2.vasp.jobs.mp import (
    MPMetaGGARelaxMaker,
    MPPreRelaxMaker,
    _get_kspacing_params,
)

expected_incar = {
    "ISIF": 3,
    "IBRION": 2,
    "NSW": 99,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LREAL": False,
    "LWAVE": False,
    "LCHARG": True,
    "EDIFF": 1e-05,
    "EDIFFG": -0.02,
    "GGA": "PS",
}


def test_mp_pre_relax_maker_default_values():
    maker = MPPreRelaxMaker()
    assert maker.name == "MP PreRelax"
    assert {*maker.input_set_generator.config_dict} >= {"INCAR", "KPOINTS", "POTCAR"}
    for key, expected in expected_incar.items():
        actual = maker.input_set_generator.config_dict["INCAR"][key]
        assert actual == expected, f"{key=}, {actual=}, {expected=}"


def test_mp_relax_maker_default_values():
    maker = MPMetaGGARelaxMaker()
    assert maker.name == "MP Relax"
    assert {*maker.input_set_generator.config_dict} >= {"INCAR", "KPOINTS", "POTCAR"}
    for key, expected in expected_incar.items():
        actual = maker.input_set_generator.config_dict["INCAR"][key]
        assert actual == expected, f"{key=}, {actual=}, {expected=}"


@pytest.mark.parametrize(
    "initial_static_maker, final_relax_maker",
    [
        (MPPreRelaxMaker(), MPMetaGGARelaxMaker()),
        (MPPreRelaxMaker(), None),
        (None, MPMetaGGARelaxMaker()),
        (None, None),  # test it doesn't raise without optional makers
    ],
)
def test_mp_meta_gga_relax_default_values(initial_static_maker, final_relax_maker):
    job = MPMetaGGARelax(
        initial_relax_maker=initial_static_maker, final_relax_maker=final_relax_maker
    )
    assert isinstance(job.initial_relax_maker, type(initial_static_maker))
    if initial_static_maker:
        assert job.initial_relax_maker.name == "MP PreRelax"

    assert isinstance(job.final_relax_maker, type(final_relax_maker))
    if final_relax_maker:
        assert job.final_relax_maker.name == "MP Relax"

    assert job.name == "MP Meta-GGA Relax"


def test_mp_meta_gga_relax_custom_values():
    initial_relax_maker = MPPreRelaxMaker()
    final_relax_maker = MPMetaGGARelaxMaker()
    job = MPMetaGGARelax(
        name="Test",
        initial_relax_maker=initial_relax_maker,
        final_relax_maker=final_relax_maker,
    )
    assert job.name == "Test"
    assert job.initial_relax_maker == initial_relax_maker
    assert job.final_relax_maker == final_relax_maker


def test_get_kspacing_params():
    bandgap = 0.05
    bandgap_tol = 0.1
    params = _get_kspacing_params(bandgap, bandgap_tol)
    assert params["KSPACING"] == 0.22

    bandgap = 1.0
    params = _get_kspacing_params(bandgap, bandgap_tol)
    assert params["KSPACING"] == approx(0.30235235)
