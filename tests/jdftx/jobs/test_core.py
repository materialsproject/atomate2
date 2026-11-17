from jobflow import run_locally

from atomate2.jdftx.jobs.core import IonicMinMaker, LatticeMinMaker, SinglePointMaker
from atomate2.jdftx.schemas.task import TaskDoc
from atomate2.jdftx.sets.core import (
    IonicMinSetGenerator,
    LatticeMinSetGenerator,
    SinglePointSetGenerator,
)


def test_sp_maker(mock_jdftx, si_structure, mock_filenames, clean_dir):
    ref_paths = {"single_point": "sp_test"}

    fake_run_jdftx_kwargs = {}

    mock_jdftx(ref_paths, fake_run_jdftx_kwargs)

    maker = SinglePointMaker(input_set_generator=SinglePointSetGenerator())
    maker.input_set_generator.user_settings["coords-type"] = "Lattice"

    job = maker.make(si_structure)

    responses = run_locally(job, create_folders=True, ensure_success=True)
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)


def test_ionicmin_maker(mock_jdftx, si_structure, mock_filenames, clean_dir):
    ref_paths = {"ionic_min": "ionicmin_test"}

    fake_run_jdftx_kwargs = {}

    mock_jdftx(ref_paths, fake_run_jdftx_kwargs)

    maker = IonicMinMaker(input_set_generator=IonicMinSetGenerator())
    maker.input_set_generator.user_settings["coords-type"] = "Lattice"

    job = maker.make(si_structure)

    responses = run_locally(job, create_folders=True, ensure_success=True)
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)


def test_latticemin_maker(mock_jdftx, si_structure, mock_filenames, clean_dir):
    ref_paths = {"lattice_min": "latticemin_test"}

    fake_run_jdftx_kwargs = {}

    mock_jdftx(ref_paths, fake_run_jdftx_kwargs)

    maker = LatticeMinMaker(input_set_generator=LatticeMinSetGenerator())
    # Need to be in Lattice coords to compare to test files
    maker.input_set_generator.user_settings["coords-type"] = "Lattice"

    job = maker.make(si_structure)

    responses = run_locally(job, create_folders=True, ensure_success=True)
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)


def test_output_loads_from_store(
    mock_jdftx, si_structure, mock_filenames, clean_dir, memory_jobstore
):
    """A later job gets this output through its reference, decoded by MontyDecoder."""
    mock_jdftx({"single_point": "sp_test"}, {})

    maker = SinglePointMaker(input_set_generator=SinglePointSetGenerator())
    maker.input_set_generator.user_settings["coords-type"] = "Lattice"

    job = maker.make(si_structure)
    run_locally(job, store=memory_jobstore, create_folders=True, ensure_success=True)

    output = job.output.resolve(memory_jobstore)
    assert isinstance(output, TaskDoc)
    assert isinstance(output.calc_outputs.parameters, dict)
