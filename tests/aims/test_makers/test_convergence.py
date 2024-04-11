"""A test for AIMS convergence maker (used for GW, for instance)"""

import os

import pytest

cwd = os.getcwd()


def test_convergence(mock_aims, tmp_path, si, species_dir):
    """A test for the convergence maker"""

    from jobflow import run_locally

    from atomate2.aims.jobs.convergence import ConvergenceMaker
    from atomate2.aims.jobs.core import StaticMaker, StaticSetGenerator
    from atomate2.aims.schemas.task import ConvergenceSummary

    # mapping from job name to directory containing test files
    ref_paths = {
        "SCF Calculation 0": "k-grid-convergence-si/static-1",
        "SCF Calculation 1": "k-grid-convergence-si/static-2",
        "SCF Calculation 2": "k-grid-convergence-si/static-3",
    }

    input_set_parameters = {"species_dir": (species_dir / "light").as_posix()}

    parameters = {
        "maker": StaticMaker(
            input_set_generator=StaticSetGenerator(user_params=input_set_parameters)
        ),
        "criterion_name": "energy_per_atom",
        "epsilon": 0.2,
        "convergence_field": "k_grid",
        "convergence_steps": [[3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]],
    }

    # settings passed to fake_run_aims
    fake_run_kwargs = {}

    # automatically use fake AIMS
    mock_aims(ref_paths, fake_run_kwargs)

    # generate job
    job = ConvergenceMaker(**parameters).make(si)

    # Run the job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(cwd)

    job_uuid = job.uuid
    while responses[job_uuid][1].replace:
        job_uuid = responses[job_uuid][1].replace.all_uuids[1]

    output = responses[job_uuid][1].output

    # validate output
    assert isinstance(output, ConvergenceSummary)
    assert output.converged
    assert output.convergence_field_value == [5, 5, 5]
    assert output.actual_epsilon == pytest.approx(0.0614287)
