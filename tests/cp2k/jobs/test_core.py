import pytest
from pytest import approx


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch, test_dir):
    settings = {
        "PMG_CP2K_DATA_DIR": f"{test_dir}/cp2k/data",
        "PMG_DEFAULT_CP2K_FUNCTIONAL": "PBE",
        "PMG_DEFAULT_CP2K_BASIS_TYPE": "DZVP-MOLOPT",
        "PMG_DEFAULT_CP2K_AUX_BASIS_TYPE": "pFIT",
    }
    monkeypatch.setattr("pymatgen.core.SETTINGS", settings)


def test_static_maker(tmp_path, mock_cp2k, si_structure, basis_and_potential):
    import os

    from jobflow import run_locally

    from atomate2.cp2k.jobs.core import StaticMaker
    from atomate2.cp2k.schemas.task import TaskDocument
    from atomate2.cp2k.sets.core import StaticSetGenerator

    # mapping from job name to directory containing test files
    ref_paths = {"static": "Si_static_test"}

    # settings passed to fake_run_cp2k; adjust these to check for certain input settings
    fake_run_cp2k_kwargs = {}

    # automatically use fake CP2K
    mock_cp2k(ref_paths, fake_run_cp2k_kwargs)

    # generate job
    maker = StaticMaker(
        input_set_generator=StaticSetGenerator(user_input_settings=basis_and_potential)
    )
    job = maker.make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == approx(-214.23651374)


def test_relax_maker(tmp_path, mock_cp2k, basis_and_potential, si_structure):
    import os

    from jobflow import run_locally

    from atomate2.cp2k.jobs.core import RelaxMaker
    from atomate2.cp2k.schemas.task import TaskDocument
    from atomate2.cp2k.sets.core import RelaxSetGenerator

    # mapping from job name to directory containing test files
    ref_paths = {"relax": "Si_double_relax/relax_1"}

    # settings passed to fake_run_cp2k; adjust these to check for certain input settings
    fake_run_cp2k_kwargs = {"transmuter": {"input_settings": []}}

    # automatically use fake CP2K
    mock_cp2k(ref_paths, fake_run_cp2k_kwargs)

    # generate job
    maker = RelaxMaker(
        input_set_generator=RelaxSetGenerator(user_input_settings=basis_and_potential)
    )
    job = maker.make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == approx(-193.39161102)
    assert len(output1.calcs_reversed[0].output.ionic_steps) == 1
    assert output1.calcs_reversed[0].output.structure.lattice.abc == approx(
        si_structure.lattice.abc
    )


def test_transmuter(tmp_path, mock_cp2k, basis_and_potential, si_structure):
    import os

    import numpy as np
    from jobflow import run_locally

    from atomate2.cp2k.jobs.core import TransmuterMaker
    from atomate2.cp2k.schemas.task import TaskDocument
    from atomate2.cp2k.sets.core import StaticSetGenerator

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
        input_set_generator=StaticSetGenerator(user_input_settings=basis_and_potential),
    ).make(si_structure)

    # run the job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == approx(-404.08231791)
    scaling_matrix = output1.transformations["history"][0]["scaling_matrix"]
    assert scaling_matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
    np.testing.assert_allclose(
        output1.structure.lattice.abc, [3.866975, 3.866975, 7.733949]
    )
