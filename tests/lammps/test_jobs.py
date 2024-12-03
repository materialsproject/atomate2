import pytest
import os
import glob
from jobflow import run_locally
from pymatgen.core import Molecule, Structure
import pandas as pd

from atomate2.lammps.jobs.core import (
    LammpsNVTMaker,
    LammpsNPTMaker,
    MinimizationMaker,
    CustomLammpsMaker,
)
from atomate2.lammps.sets.core import (LammpsNVTSet, LammpsNPTSet, LammpsMinimizeSet)
from atomate2.lammps.schemas.task import LammpsTaskDocument
from atomate2.lammps.schemas.task import StoreTrajectoryOption


test_data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test_data', 'lammps'))
clean_files = ["*.lammps", "*.dump", "*.restart", "*.log", "*.data", "trajectory*"]

def test_nvt_maker(si_structure, tmp_path, test_si_force_field, mock_lammps):
    ref_paths = {'nvt_test': 'nvt_test'}
    
    fake_run_lammps_kwargs = {}
    
    mock_lammps(ref_paths, fake_run_lammps_kwargs=fake_run_lammps_kwargs)
    
    maker = LammpsNVTMaker(force_field=test_si_force_field, task_document_kwargs={'store_trajectory': StoreTrajectoryOption.PARTIAL})
    maker.name = 'nvt_test'
    
    assert maker.input_set_generator.settings['ensemble'] == 'nvt'
    
    job = maker.make(si_structure)
    
    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(os.getcwd())
    output = responses[job.uuid][1].output
    
    assert isinstance(output, LammpsTaskDocument)
    assert output.structure.volume == pytest.approx(si_structure.volume)
    assert len(output.trajectory[0]) == 11
    assert len(list(output.dump_files.keys())) == 1
    dump_key = list(output.dump_files.keys())[0]
    assert dump_key.endswith('.dump')
    assert isinstance(output.dump_files[dump_key], str)
    
def test_npt_maker(si_structure, tmp_path, test_si_force_field, mock_lammps):
    
    ref_paths = {'npt_test': 'npt_test'}
    
    fake_run_lammps_kwargs = {}
    
    mock_lammps(ref_paths, fake_run_lammps_kwargs=fake_run_lammps_kwargs)
    
    maker = LammpsNPTMaker(force_field=test_si_force_field, task_document_kwargs={'store_trajectory': StoreTrajectoryOption.PARTIAL})
    maker.name = 'npt_test'
    job = maker.make(si_structure)

    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(os.getcwd())
    output = responses[job.uuid][1].output
    
    assert isinstance(output, LammpsTaskDocument)
    assert output.structure.volume > si_structure.volume
    assert len(output.trajectory[0]) == 11
    assert len(output.dump_files.keys()) == 1
    dump_key = list(output.dump_files.keys())[0]
    assert dump_key.endswith('.dump')
    assert isinstance(output.dump_files[dump_key], str)

def test_nvt_ff_from_set(si_structure, tmp_path, test_si_force_field, mock_lammps):
    
    ref_paths = {'nvt_test': 'nvt_test'}
    
    fake_run_lammps_kwargs = {}
    
    mock_lammps(ref_paths, fake_run_lammps_kwargs=fake_run_lammps_kwargs)
    
    input_set = LammpsNVTSet(force_field=test_si_force_field)
    maker = LammpsNVTMaker(input_set_generator=input_set, task_document_kwargs={'store_trajectory': StoreTrajectoryOption.PARTIAL})
    maker.name = 'nvt_test'
    
    job = maker.make(si_structure)
    
    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(os.getcwd())
    output = responses[job.uuid][1].output
    
    assert isinstance(output, LammpsTaskDocument)
    assert isinstance(output.thermo_log[0], pd.DataFrame)
    assert isinstance(output.raw_log_file, str)
    assert len(output.thermo_log[0]) == 11
    assert output.structure.volume == pytest.approx(si_structure.volume)
    assert len(output.trajectory[0]) == 11
    assert len(output.dump_files.keys()) == 1
    dump_key = list(output.dump_files.keys())[0]
    assert dump_key.endswith('.dump')
    assert isinstance(output.dump_files[dump_key], str)
            
def test_minimization_maker(si_structure, tmp_path, test_si_force_field, mock_lammps):
    
    ref_paths = {'min_test': 'min_test'}
    
    fake_run_lammps_kwargs = {}
    
    mock_lammps(ref_paths, fake_run_lammps_kwargs=fake_run_lammps_kwargs)
    
    maker = MinimizationMaker(force_field=test_si_force_field)
    maker.name = 'min_test'
    job = maker.make(si_structure)
    
    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(os.getcwd())
    output = responses[job.uuid][1].output
    
    assert isinstance(output, LammpsTaskDocument)
    assert len(output.dump_files.keys()) == 1
    dump_key = list(output.dump_files.keys())[0]
    assert dump_key.endswith('.dump')
    assert isinstance(output.dump_files[dump_key], str)

def dummy(si_structure, tmp_path, test_si_force_field, mock_lammps):
    ref_paths = {'nvt_test': 'nvt_test'}
    
    fake_run_lammps_kwargs = {}
    
    mock_lammps(ref_paths, fake_run_lammps_kwargs=fake_run_lammps_kwargs)
    
    maker = LammpsNVTMaker(force_field=test_si_force_field, task_document_kwargs={'store_trajectory': StoreTrajectoryOption.PARTIAL})
    maker.name = 'nvt_test'
    
    assert maker.input_set_generator.settings['ensemble'] == 'nvt'
    
    job = maker.make(si_structure)
    
    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(os.getcwd())
    output = responses[job.uuid][1].output
    
    assert isinstance(output, LammpsTaskDocument)
    assert isinstance(output.thermo_log[0], pd.DataFrame)
    assert isinstance(output.raw_log_file, str)
    assert len(output.thermo_log[0]) == 11
    assert output.structure.volume == pytest.approx(si_structure.volume)
    assert len(output.trajectory[0]) == 11
    assert len(list(output.dump_files.keys())) == 1
    dump_key = list(output.dump_files.keys())[0]
    assert dump_key.endswith('.dump')
    assert isinstance(output.dump_files[dump_key], str)
    for f in clean_files:
        for file in glob.glob(f):
            os.remove(file)
            