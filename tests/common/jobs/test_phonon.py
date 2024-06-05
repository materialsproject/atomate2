from jobflow import run_locally
from numpy.testing import assert_allclose
from pymatgen.core.structure import Structure
from atomate2.common.jobs.phonons import get_supercell_size


def test_phonon_get_supercell_size(clean_dir, si_structure: Structure):
    job = get_supercell_size(si_structure, min_length=18, prefer_90_degrees=True)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    assert_allclose(responses[job.uuid][1].output, [[6, -2, 0], [0, 6, 0], [-3, -2, 5]])
