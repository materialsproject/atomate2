import warnings

import numpy as np
import pytest
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


def test_generate_frequencies_eigenvectors_magmom(clean_dir, si_structure, monkeypatch):
    from phonopy import Phonopy
    from pymatgen.io.phonopy import get_phonopy_structure

    from atomate2.common.jobs import phonons

    kwargs = {
        "supercell_matrix": np.eye(3) * 2,
        "displacement": 0.01,
        "sym_reduce": True,
        "symprec": 1e-4,
        "use_symmetrized_structure": None,
        "kpath_scheme": "seekpath",
        "code": "vasp",
    }
    run_info = {
        f"{run}_run_{key}": None
        for run in ("static", "born", "optimization")
        for key in ("job_dir", "uuid")
    }
    magnetic = si_structure.copy(site_properties={"magmom": [1, -1]})
    supercells = phonons.generate_phonon_displacements.original(
        structure=magnetic, **kwargs
    )
    ideal = Phonopy(get_phonopy_structure(si_structure), np.eye(3) * 2).supercell
    forces = [-10 * (cell.cart_coords - ideal.positions) for cell in supercells]
    data = {"forces": forces, "uuids": ["1"], "dirs": ["1"]}

    # the atomate2 document computes the phonons, so the job must not
    def generate_phonon_object(*_, **__):
        pytest.fail("The phonons are computed twice.")

    monkeypatch.setattr(phonons, "_generate_phonon_object", generate_phonon_object)
    docs = [
        phonons.generate_frequencies_eigenvectors.original(
            structure=structure,
            displacement_data=data,
            total_dft_energy=-10.0,
            store_force_constants=False,
            **kwargs,
            **run_info,
        )
        for structure in (si_structure, magnetic)
    ]

    assert "magmom" not in docs[1].structure.site_properties
    assert docs[1].structure == docs[0].structure
    assert_allclose(docs[1].free_energies, docs[0].free_energies)


@pytest.mark.parametrize("verbose", [True, False])
def test_get_displacements_warns_magmom(clean_dir, si_structure, verbose):
    from atomate2.vasp.flows.phonons import PhononMaker

    magnetic = si_structure.copy(site_properties={"magmom": [1, -1]})
    job = PhononMaker(verbose=verbose).get_displacements(magnetic, np.eye(3) * 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_locally(job, create_folders=True, ensure_success=True)
    assert any("magnetic moments" in str(w.message) for w in caught) == verbose
