from pytest import approx


def test_static_maker(mock_cp2k, si_structure, clean_dir, basis_and_potential):
    from jobflow import run_locally
    from atomate2.cp2k.jobs.core import StaticMaker
    from atomate2.cp2k.schemas.task import TaskDocument

    # mapping from job name to directory containing test files
    ref_paths = {"static": "Si_static_test"}

    # settings passed to fake_run_cp2k; adjust these to check for certain input settings
    fake_run_cp2k_kwargs = {}

    # automatically use fake CP2K
    mock_cp2k(ref_paths, fake_run_cp2k_kwargs)

    # generate job
    job = StaticMaker().make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == approx(-214.23651374775685)

def test_relax_maker(mock_cp2k, clean_dir, si_structure):
    from jobflow import run_locally
    from atomate2.cp2k.jobs.core import RelaxMaker
    from atomate2.cp2k.schemas.task import TaskDocument

    # mapping from job name to directory containing test files
    ref_paths = {"relax": "Si_double_relax/relax_1"}

    # settings passed to fake_run_cp2k; adjust these to check for certain input settings
    fake_run_cp2k_kwargs = {"transmuter": {"input_settings": []}}

    # automatically use fake CP2K
    mock_cp2k(ref_paths, fake_run_cp2k_kwargs)

    # generate job
    job = RelaxMaker().make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == approx(-193.39161102270234)
    assert len(output1.calcs_reversed[0].output.ionic_steps) == 1

def test_transmuter(mock_cp2k, clean_dir, si_structure):
    import numpy as np
    from jobflow import run_locally

    from atomate2.cp2k.jobs.core import TransmuterMaker
    from atomate2.cp2k.schemas.task import TaskDocument

    # mapping from job name to directory containing test files
    ref_paths = {"transmuter": "Si_transmuter"}

    # settings passed to fake_run_cp2k; adjust these to check for certain settings
    fake_run_cp2k_kwargs = {"transmuter": {"input_settings": []}}

    # automatically use fake CP2K and write POTCAR.spec during the test
    mock_cp2k(ref_paths, fake_run_cp2k_kwargs)

    # generate transmuter job
    job = TransmuterMaker(
        transformations=["SupercellTransformation"],
        transformation_params=[{"scaling_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 2))}],
    ).make(si_structure)

    # run the job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == approx(-404.0823179177859)
    assert output1.transformations["history"][0]["scaling_matrix"] == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 2],
    ]
    np.testing.assert_allclose(
        output1.structure.lattice.abc, [3.866975, 3.866975, 7.733949]
    )
