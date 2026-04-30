from jobflow import run_locally
from numpy.testing import assert_allclose
from pymatgen.core import Structure

from atomate2.common.jobs.phonons import get_supercell_size


def test_supercell(si_structure, tmp_dir):
    supercell = get_supercell_size(
        si_structure, min_length=10, max_length=None, prefer_90_degrees=False
    )

    responses = run_locally(supercell, create_folders=True, ensure_success=True)
    assert_allclose(
        responses[supercell.output.uuid][1].output, [[3, -1, 0], [0, 4, 0], [-2, -1, 3]]
    )


def test_supercell1(si_structure, tmp_dir):
    supercell = get_supercell_size(
        si_structure, min_length=8, max_length=13, prefer_90_degrees=True
    )

    responses = run_locally(supercell, create_folders=True, ensure_success=True)
    assert_allclose(
        responses[supercell.output.uuid][1].output, [[3, -1, 0], [0, 3, 0], [-1, -1, 3]]
    )


def test_supercell2(si_structure, tmp_dir):
    supercell = get_supercell_size(
        si_structure, min_length=8, max_length=20, prefer_90_degrees=True
    )

    responses = run_locally(supercell, create_folders=True, ensure_success=True)
    assert_allclose(
        responses[supercell.output.uuid][1].output, [[6, -2, 0], [0, 6, 0], [-3, -2, 5]]
    )


def test_phonon_get_supercell_size(clean_dir, si_structure: Structure):
    job = get_supercell_size(
        si_structure, min_length=18, max_length=25, prefer_90_degrees=True
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    assert_allclose(responses[job.uuid][1].output, [[6, -2, 0], [0, 6, 0], [-3, -2, 5]])


def test_supercell_orthorhombic(clean_dir, si_structure: Structure):
    job1 = get_supercell_size(
        si_structure,
        min_length=5,
        max_length=10,
        prefer_90_degrees=False,
        allow_orthorhombic=True,
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job1, create_folders=True, ensure_success=True)

    assert_allclose(
        responses[job1.uuid][1].output, [[2, -1, 0], [0, 2, 0], [-1, -1, 2]]
    )

    job2 = get_supercell_size(
        si_structure,
        min_length=5,
        max_length=10,
        prefer_90_degrees=True,
        allow_orthorhombic=True,
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job2, create_folders=True, ensure_success=True)

    assert_allclose(
        responses[job2.uuid][1].output, [[2, -1, 0], [0, 2, 0], [-1, -1, 2]]
    )
