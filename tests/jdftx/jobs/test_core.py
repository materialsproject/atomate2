import pytest
from unittest.mock import patch
from jobflow import run_locally

from atomate2.jdftx.jobs.core import RelaxMaker
from atomate2.jdftx.sets.core import RelaxSetGenerator
from atomate2.jdftx.jobs.base import BaseJdftxMaker

#@patch('atomate2.jdftx.jobs.base.BaseJdftxMaker.make')
#@patch('atomate2.jdftx.jobs.base.write_jdftx_input_set')
def test_static_maker(mock_jdftx, si_structure):
    import os


    ref_paths = {}

    fake_run_jdftx_kwargs = {}

    mock_jdftx(ref_paths, fake_run_jdftx_kwargs) #updates _REF_PATHS and _FAKE_RUN_JDFTX_KWARGS & monkeypatches
    #run_jdftx and get_input_set

    maker = RelaxMaker(input_set_generator=RelaxSetGenerator())

    job = maker.make(si_structure)

 #   MockMake.assert_called_once()
 #   MockWrite.assert_called_once()


