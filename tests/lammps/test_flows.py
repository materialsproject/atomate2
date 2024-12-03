import pytest
import os
from jobflow import run_locally
import pandas as pd

from atomate2.lammps.flows.core import MeltQuenchThermalizeMaker
from atomate2.lammps.jobs.core import LammpsNPTMaker
from atomate2.lammps.schemas.task import LammpsTaskDocument

def test_melt_quench_thermalize_maker(si_structure, tmp_path, test_si_force_field, mock_lammps):
    ref_paths = {'melt_test': 'meltquenchtherm_test/melt_test',
                 'quench_test': 'meltquenchtherm_test/quench_test',
                 'thermalize_test': 'meltquenchtherm_test/therm_test'}
    
    fake_run_lammps_kwargs = {}
    
    mock_lammps(ref_paths, fake_run_lammps_kwargs=fake_run_lammps_kwargs)
    
    npt = LammpsNPTMaker(force_field=test_si_force_field)
    maker = MeltQuenchThermalizeMaker.from_temperature_steps(npt_maker=npt, nvt_maker=None)
    maker.name = 'melt_quench_thermalize_test'
    maker.melt_maker.name = 'melt_test'
    maker.quench_maker.name = 'quench_test'
    maker.thermalize_maker.name = 'thermalize_test'
    
    job = maker.make(si_structure)
    
    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(os.getcwd())
    outputs = [responses[job[i].uuid][1].output for i in range(3)]
    
    for output in outputs:
        assert isinstance(output, LammpsTaskDocument)
        assert len(output.dump_files.keys()) == 1
        dump_key = list(output.dump_files.keys())[0]
        assert dump_key.endswith('.dump')
        assert isinstance(output.dump_files[dump_key], str)
        assert (output.structure.volume - si_structure.volume)/si_structure.volume < 0.1
    