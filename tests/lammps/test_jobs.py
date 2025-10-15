import os

import pytest
from jobflow import run_locally

from atomate2.lammps.jobs.core import LammpsNPTMaker, LammpsNVTMaker, MinimizationMaker
from atomate2.lammps.schemas.task import LammpsTaskDocument, StoreTrajectoryOption
from atomate2.lammps.sets.core import LammpsNVTSet


def test_nvt_maker(si_structure, tmp_path, test_si_force_field, mock_lammps):
    ref_paths = {"nvt_test": "nvt_test"}

    fake_run_lammps_kwargs = {}

    mock_lammps(ref_paths, fake_run_lammps_kwargs=fake_run_lammps_kwargs)

    generator = LammpsNVTSet(
        settings={
            "start_temp": 300,
            "end_temp": 1000,
            "friction": 0.1,
            "nsteps": 100000,
            "timestep": 0.001,
            "log_interval": 500,
        }
    )
    maker = LammpsNVTMaker(
        force_field=test_si_force_field,
        input_set_generator=generator,
        task_document_kwargs={"store_trajectory": StoreTrajectoryOption.PARTIAL},
    )
    maker.name = "nvt_test"

    if isinstance(maker.input_set_generator.settings, dict):
        assert maker.input_set_generator.settings["ensemble"] == "nvt"
    else:
        assert maker.input_set_generator.settings.ensemble == "nvt"

    supercell = si_structure.make_supercell([5, 5, 5])
    job = maker.make(supercell)

    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(os.getcwd())
    output = responses[job.uuid][1].output

    assert isinstance(output, LammpsTaskDocument)
    assert output.structure.volume == pytest.approx(supercell.volume)
    assert len(list(output.dump_files.keys())) == 1
    dump_key = next(iter(output.dump_files.keys()))
    assert dump_key.endswith(".dump")
    assert isinstance(output.dump_files[dump_key], str)


def test_npt_maker(si_structure, tmp_path, test_si_force_field, mock_lammps):
    ref_paths = {"npt_test": "npt_test"}

    fake_run_lammps_kwargs = {}

    mock_lammps(ref_paths, fake_run_lammps_kwargs=fake_run_lammps_kwargs)

    maker = LammpsNPTMaker(force_field=test_si_force_field)
    maker.name = "npt_test"
    job = maker.make(si_structure.make_supercell([5, 5, 5]))

    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(os.getcwd())
    output = responses[job.uuid][1].output

    assert isinstance(output, LammpsTaskDocument)
    assert len(output.dump_files.keys()) == 1
    dump_key = next(iter(output.dump_files.keys()))
    assert dump_key.endswith(".dump")
    assert isinstance(output.dump_files[dump_key], str)


def test_minimization_maker(si_structure, tmp_path, test_si_force_field, mock_lammps):
    ref_paths = {"min_test": "min_test"}

    fake_run_lammps_kwargs = {}

    mock_lammps(ref_paths, fake_run_lammps_kwargs=fake_run_lammps_kwargs)

    maker = MinimizationMaker(force_field=test_si_force_field)
    maker.input_set_generator.update_settings({"nsteps": 1000})
    maker.name = "min_test"
    supercell = si_structure.make_supercell([5, 5, 5])
    job = maker.make(supercell)

    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(os.getcwd())
    output = responses[job.uuid][1].output

    assert isinstance(output, LammpsTaskDocument)
    assert len(output.dump_files.keys()) == 1
    dump_key = next(iter(output.dump_files.keys()))
    assert dump_key.endswith(".dump")
    assert isinstance(output.dump_files[dump_key], str)
    assert list(output.thermo_log[0]["PotEng"])[-1] == pytest.approx(
        -327.96091, abs=1e-3
    ), "Final potential energy does not match expected value."
