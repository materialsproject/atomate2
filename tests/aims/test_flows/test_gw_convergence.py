"""A test for GW workflows for FHI-aims."""

import pytest

# from atomate2.aims.utils.msonable_atoms import MSONableAtoms


def test_gw_convergence(mock_aims, tmp_dir, o2, species_dir):
    """A test for the GW convergence maker for molecule with respect to the basis set
    size
    """

    from jobflow import run_locally
    from pymatgen.io.aims.sets.bs import GWSetGenerator

    from atomate2.aims.flows.gw import GWConvergenceMaker
    from atomate2.aims.jobs.core import GWMaker
    from atomate2.aims.schemas.task import ConvergenceSummary

    # mapping from job name to directory containing test files
    ref_paths = {
        "GW 0": "basis-gw-convergence-o2/static-1",
        "GW 1": "basis-gw-convergence-o2/static-2",
    }

    input_set_parameters = {}

    parameters = {
        "maker": GWMaker(
            input_set_generator=GWSetGenerator(user_params=input_set_parameters)
        ),
        "criterion_name": "vbm",
        "epsilon": 0.05,
        "convergence_field": "species_dir",
        "convergence_steps": [
            (species_dir / "light").as_posix(),
            (species_dir / "tight").as_posix(),
        ],
    }

    # settings passed to fake_run_aims
    fake_run_kwargs = {}

    # automatically use fake AIMS
    mock_aims(ref_paths, fake_run_kwargs)

    # generate job
    job = GWConvergenceMaker(**parameters).make(o2)

    # Run the job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    job_uuid = job.uuid
    while responses[job_uuid][1].replace:
        job_uuid = responses[job_uuid][1].replace.all_uuids[1]

    output = responses[job_uuid][1].output

    # validate output
    assert isinstance(output, ConvergenceSummary)
    assert output.converged
    assert "tight" in output.convergence_field_value
    assert output.actual_epsilon == pytest.approx(0.04594407)
