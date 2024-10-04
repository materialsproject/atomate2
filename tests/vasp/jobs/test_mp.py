from collections.abc import Callable
from pathlib import Path

import pytest
from emmet.core.tasks import TaskDoc
from jobflow import run_locally
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPScanRelaxSet

from atomate2.vasp.jobs.mp import (
    MPGGARelaxMaker,
    MPGGAStaticMaker,
    MPMetaGGARelaxMaker,
    MPMetaGGAStaticMaker,
    MPPreRelaxMaker,
)

expected_incar = {
    "ISIF": 3,
    "IBRION": 2,
    "NSW": 99,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LREAL": "Auto",
    "LWAVE": False,
    "LCHARG": True,
    "EDIFF": 1e-05,
    "EDIFFG": -0.02,
}


def test_mp_pre_relax_maker_default_values():
    maker = MPPreRelaxMaker()
    assert maker.name == "MP pre-relax"
    assert {*maker.input_set_generator.config_dict} >= {"INCAR", "POTCAR"}
    for key, expected in expected_incar.items():
        actual = maker.input_set_generator.config_dict["INCAR"][key]
        assert actual == expected, f"{key=}, {actual=}, {expected=}"
    assert maker.input_set_generator.inherit_incar is False


def test_mp_meta_gga_relax_maker_default_values():
    maker = MPMetaGGARelaxMaker()
    assert maker.name == "MP meta-GGA relax"
    assert {*maker.input_set_generator.config_dict} >= {"INCAR", "POTCAR"}
    for key, expected in expected_incar.items():
        actual = maker.input_set_generator.config_dict["INCAR"][key]
        assert actual == expected, f"{key=}, {actual=}, {expected=}"
    assert maker.input_set_generator.inherit_incar is False


def test_mp_meta_gga_static_maker(
    mock_vasp: Callable, clean_dir: None, vasp_test_dir: Path
):
    # map from job name to directory containing reference input/output files
    ref_paths = {
        "MP meta-GGA static": "Si_mp_meta_gga_relax/r2scan_final_static",
    }
    si_struct = Structure.from_file(
        f"{vasp_test_dir}/Si_mp_meta_gga_relax/r2scan_final_static/inputs/POSCAR.gz"
    )

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {key: {"incar_settings": []} for key in ref_paths}

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    input_set_gen = MPScanRelaxSet(
        bandgap=0.8249, user_incar_settings={"LWAVE": True, "LCHARG": True}
    )
    maker = MPMetaGGAStaticMaker(input_set_generator=input_set_gen)
    job = maker.make(si_struct)

    # ensure flow runs successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate output
    task_doc = responses[job.uuid][1].output
    assert isinstance(task_doc, TaskDoc)
    assert task_doc.output.energy == pytest.approx(-46.8613738)
    assert maker.input_set_generator.inherit_incar is False


def test_mp_meta_gga_relax_maker(
    mock_vasp: Callable, clean_dir: None, vasp_test_dir: Path
):
    # map from job name to directory containing reference input/output files
    ref_paths = {
        "MP meta-GGA relax": "Si_mp_meta_gga_relax/r2scan_relax",
    }
    si_struct = Structure.from_file(
        f"{vasp_test_dir}/Si_mp_meta_gga_relax/r2scan_final_static/inputs/POSCAR.gz"
    )

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        key: {"incar_settings": ["LWAVE", "LCHARG"]} for key in ref_paths
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    input_set_gen = MPScanRelaxSet(
        bandgap=0.4786, user_incar_settings={"LWAVE": True, "LCHARG": True}
    )
    maker = MPMetaGGARelaxMaker(input_set_generator=input_set_gen)
    job = maker.make(si_struct)

    # ensure flow runs successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate output
    task_doc = responses[job.uuid][1].output
    assert isinstance(task_doc, TaskDoc)
    assert task_doc.output.energy == pytest.approx(-46.86703814)
    assert maker.input_set_generator.inherit_incar is False


def test_mp_gga_relax_maker(mock_vasp: Callable, clean_dir: None, vasp_test_dir: Path):
    # map from job name to directory containing reference input/output files
    ref_paths = {
        "MP GGA relax": "Si_mp_gga_relax/GGA_Relax_1",
    }
    si_struct = Structure.from_file(
        f"{vasp_test_dir}/Si_mp_gga_relax/GGA_Relax_1/inputs/POSCAR.gz"
    )

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {key: {"incar_settings": ["LWAVE"]} for key in ref_paths}

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    maker = MPGGARelaxMaker()
    job = maker.make(si_struct)

    # ensure flow runs successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate output
    task_doc = responses[job.uuid][1].output
    assert isinstance(task_doc, TaskDoc)
    assert task_doc.output.energy == pytest.approx(-10.84140641)
    assert maker.input_set_generator.inherit_incar is False


# TODO ideally should have a set of ref files and a full test for an MP static calc
def test_mp_gga_static_maker_prev_dir(vasp_test_dir: Path):
    prev_dir = f"{vasp_test_dir}/Si_hse_band_structure/hse_static/outputs"
    structure = Structure.from_file(f"{prev_dir}/POSCAR.gz")

    maker = MPGGAStaticMaker()
    input_set = maker.input_set_generator.get_input_set(
        structure=structure, potcar_spec=True, prev_dir=prev_dir
    )
    assert input_set.incar["ENCUT"] == 520
    assert maker.input_set_generator.inherit_incar is False
