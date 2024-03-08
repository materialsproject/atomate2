import numpy as np
import pytest
from jobflow import run_locally
from pymatgen.analysis.elasticity import Stress
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2 import SETTINGS
from atomate2.common.jobs.elastic import generate_elastic_deformations
from atomate2.common.schemas.elastic import expand_strains


@pytest.mark.parametrize("conventional", [False, True])
def test_reduce_expand_strains(clean_dir, symmetry_structure, conventional):
    """Test the reduced and expanded strains are the same."""

    if conventional:
        spg = SpacegroupAnalyzer(symmetry_structure, symprec=SETTINGS.SYMPREC)
        structure = spg.get_conventional_standard_structure()
    else:
        structure = symmetry_structure

    full_strains = _get_strains(structure, sym_reduce=False)
    reduced_strains = _get_strains(structure, sym_reduce=True)

    dummy_stresses = [Stress(np.zeros((3, 3)))] * len(reduced_strains)
    dummy = ["dummy"] * len(reduced_strains)

    recovered_strains, _, _, _ = expand_strains(
        structure,
        reduced_strains,
        stresses=dummy_stresses,
        uuids=dummy,
        job_dirs=dummy,
        symprec=SETTINGS.SYMPREC,
    )

    assert len(full_strains) == len(recovered_strains)

    for fs in full_strains:
        assert any(np.allclose(fs, rs) for rs in recovered_strains)


def _get_strains(structure, sym_reduce):
    """Get applied strains to deform the deformed structures."""

    job = generate_elastic_deformations(
        structure,
        order=2,
        symprec=SETTINGS.SYMPREC,
        sym_reduce=sym_reduce,
    )

    response = run_locally(job, ensure_success=True)
    deformations = response[job.uuid][1].output

    return [d.green_lagrange_strain for d in deformations]
