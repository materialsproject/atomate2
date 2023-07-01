import numpy as np
import pytest
from jobflow import run_locally
from pymatgen.analysis.elasticity import Stress
from pymatgen.core import Structure

from atomate2 import SETTINGS
from atomate2.common.schemas.elastic import _expand_strains
from atomate2.vasp.jobs.elastic import generate_elastic_deformations


@pytest.mark.parametrize("conventional", [False, True])
def test_reduce_expand_deformation(clean_dir, si_structure, conventional):
    """Test reduced and expanded strains are the same all space groups."""

    # can use any structure to get all the strains
    full_strains = _get_strains(si_structure, conventional, sym_reduce=False)

    structures = _get_prototype_structures()

    for structure in structures:
        _test_one_structure(structure, full_strains, conventional)


def _test_one_structure(structure: Structure, full_strains, conventional):
    """Test reduced and expanded strains are the same for one structure."""

    reduced_strains = _get_strains(structure, conventional, sym_reduce=True)

    dummy_stresses = [Stress(np.zeros((3, 3)))] * len(reduced_strains)
    dummy = ["dummy"] * len(reduced_strains)

    recovered_strains, _, _, _ = _expand_strains(
        structure,
        reduced_strains,
        stresses=dummy_stresses,
        uuids=dummy,
        job_dirs=dummy,
        symprec=SETTINGS.SYMPREC,
    )

    assert len(full_strains) == len(recovered_strains)

    # TODO, this can be slow
    for fs in full_strains:
        assert any(np.allclose(fs, rs) for rs in recovered_strains)


def _get_strains(structure, conventional, sym_reduce):
    """Return a list of strains associated with the deformed structures."""

    job = generate_elastic_deformations(
        structure,
        order=2,
        conventional=conventional,
        symprec=SETTINGS.SYMPREC,
        sym_reduce=sym_reduce,
    )

    response = run_locally(job, ensure_success=True)
    deformations = response[job.uuid][1].output
    return [d.green_lagrange_strain for d in deformations]


def _get_prototype_structures():
    """Return a list of prototype structures, one from each of the 230 space groups."""

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    return [structure]
