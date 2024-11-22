from jobflow import run_locally
from numpy.testing import assert_allclose

from atomate2.common.jobs.phonons import get_supercell_size


def test_supercell(si_structure, tmp_dir):
    super = get_supercell_size(
        si_structure, min_length=10, max_length=None, prefer_90_degrees=False
    )

    responses = run_locally(super, create_folders=True, ensure_success=True)
    assert_allclose(
        responses[super.output.uuid][1].output, [[3, -1, 0], [0, 4, 0], [-2, -1, 3]]
    )


def test_supercell1(si_structure, tmp_dir):
    super = get_supercell_size(
        si_structure, min_length=8, max_length=13, prefer_90_degrees=True
    )

    responses = run_locally(super, create_folders=True, ensure_success=True)
    assert_allclose(
        responses[super.output.uuid][1].output, [[3, -1, 0], [0, 3, 0], [-1, -1, 3]]
    )


def test_supercell2(si_structure, tmp_dir):
    super = get_supercell_size(
        si_structure, min_length=8, max_length=20, prefer_90_degrees=True
    )

    responses = run_locally(super, create_folders=True, ensure_success=True)
    assert_allclose(
        responses[super.output.uuid][1].output, [[6, -2, 0], [0, 6, 0], [-3, -2, 5]]
    )
