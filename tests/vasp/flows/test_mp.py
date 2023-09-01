import pytest
from pymatgen.core import Structure

from atomate2.vasp.flows.mp import MPGGADoubleRelaxMaker, MPMetaGGARelax
from atomate2.vasp.jobs.mp import (
    MPMetaGGARelaxMaker,
    MPPreRelaxMaker,
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
    assert maker.name == "MP pre-relax"
    assert {*maker.input_set_generator.config_dict} >= {"INCAR", "KPOINTS", "POTCAR"}
    for key, expected in expected_incar.items():
        actual = maker.input_set_generator.config_dict["INCAR"][key]
        assert actual == expected, f"{key=}, {actual=}, {expected=}"


def test_mp_relax_maker_default_values():
    maker = MPMetaGGARelaxMaker()
    assert maker.name == "MP meta-GGA Relax"
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
        initial_maker=initial_static_maker, final_relax_maker=final_relax_maker
    )
    assert isinstance(job.initial_maker, type(initial_static_maker))
    if initial_static_maker:
        assert job.initial_maker.name == "MP pre-relax"

    assert isinstance(job.final_relax_maker, type(final_relax_maker))
    if final_relax_maker:
        assert job.final_relax_maker.name == "MP meta-GGA Relax"

    assert job.name == "MP Meta-GGA Relax"


def test_mp_meta_gga_relax_custom_values():
    initial_maker = MPPreRelaxMaker()
    final_relax_maker = MPMetaGGARelaxMaker()
    job = MPMetaGGARelax(
        name="Test",
        initial_maker=initial_maker,
        final_relax_maker=final_relax_maker,
    )
    assert job.name == "Test"
    assert job.initial_maker == initial_maker
    assert job.final_relax_maker == final_relax_maker


def test_mp_meta_gga_relax(mock_vasp, clean_dir, vasp_test_dir):
    from emmet.core.tasks import TaskDoc
    from jobflow import run_locally

    # map from job name to directory containing reference output files
    pre_relax_dir = "Si_mp_metagga_relax/pbesol_pre_relax"
    ref_paths = {
        "MP pre-relax": pre_relax_dir,
        "MP meta-GGA relax": "Si_mp_metagga_relax/r2scan_relax",
        "MP meta-GGA static": "Si_mp_metagga_relax/r2scan_final_static",
    }
    si_struct = Structure.from_file(f"{vasp_test_dir}/{pre_relax_dir}/inputs/POSCAR")

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        key: {"incar_settings": ["LWAVE", "LCHARG"]} for key in ref_paths
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate flow
    flow = MPGGADoubleRelaxMaker().make(si_struct)

    # ensure flow runs successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate output
    output = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(output, TaskDoc)
    assert output.output.energy == pytest.approx(-10.85043620)
