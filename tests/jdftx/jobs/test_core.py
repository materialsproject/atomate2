import pytest

def test_static_maker(mock_jdftx, si_structure):
    import os

    from atomate2.jdftx.jobs.core import RelaxMaker
    from atomate2.jdftx.sets.core import RelaxSetGenerator


    ref_paths = {}

    fake_run_jdftx_kwargs = {}

    mock_jdftx(ref_paths, fake_run_jdftx_kwargs) #updates _REF_PATHS and _FAKE_RUN_JDFTX_KWARGS & monkeypatches
    #run_jdftx and get_input_set

    maker = RelaxMaker(input_set_generator=RelaxSetGenerator())

    job = maker.make(si_structure)

