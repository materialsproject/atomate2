from jobflow import run_locally
from numpy.testing import assert_allclose
from pymatgen.core import Lattice, Structure

from atomate2.common.jobs.phonons import (
    _generate_phonon_object,
    _get_num_harmonic_supercells,
    get_supercell_size,
)


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


def test_get_num_harmonic_supercells():
    # a distorted Cu cell needs more than three finite displacements
    structure = Structure(
        Lattice.orthorhombic(3.6, 3.7, 3.8),
        ["Cu"] * 4,
        [[0, 0, 0], [0.5, 0.5, 0.02], [0.5, 0.03, 0.5], [0.01, 0.5, 0.5]],
    )
    phonon = _generate_phonon_object(
        structure,
        supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        displacement=0.01,
        sym_reduce=True,
        symprec=1e-3,
        use_symmetrized_structure=None,
        kpath_scheme="seekpath",
        code="vasp",
    )
    # the hiPhive flow counts the free force constants with ALM
    assert _get_num_harmonic_supercells(phonon, 0) == 14
    # the pheasy flow passes its own count, 96 free force constants for 32 atoms
    assert _get_num_harmonic_supercells(phonon, 0, count_free_fcs=lambda: 96) == 3
    # a number set by the user is used as it is
    assert _get_num_harmonic_supercells(phonon, 5) == 5
