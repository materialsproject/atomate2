import numpy as np
import pytest
from jobflow import run_locally
from pymatgen.analysis.elasticity import Stress
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2 import SETTINGS
from atomate2.common.jobs.elastic import (
    fit_elastic_tensor,
    generate_elastic_deformations,
)
from atomate2.common.schemas.elastic import ElasticWarnings, expand_strains


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


def test_fit_elastic_tensor(clean_dir, si_structure, caplog):
    conventional = SpacegroupAnalyzer(
        si_structure
    ).get_conventional_standard_structure()

    deformation_data = [
        {
            "stress": [
                [15.73376749, 0.0, 0.0],
                [0.0, 6.40261126, 0.0],
                [0.0, 0.0, 6.40261126],
            ],
            "deformation": [
                [0.9899494936611666, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "uuid": "b7715382-9130-409c-ae2d-32a01321a0d0",
            "job_dir": "a",
        },
        {
            "stress": [
                [7.74111679, 0.0, 0.0],
                [0.0, 3.05807413, -0.0],
                [0.0, -0.0, 3.05807413],
            ],
            "deformation": [
                [0.99498743710662, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "uuid": "6cd2688c-764b-4a08-80f8-5c3ed75b91b9",
            "job_dir": "b",
        },
        {
            "stress": [
                [-7.9262828, 0.0, -0.0],
                [0.0, -3.20998817, 0.0],
                [0.0, 0.0, -3.20998817],
            ],
            "deformation": [
                [1.004987562112089, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "uuid": "fc3405d7-4171-4fe6-ab1b-086378ae6d0f",
            "job_dir": "c",
        },
        {
            "stress": [
                [-15.60955466, 0.0, -0.0],
                [0.0, -6.14725418, 0.0],
                [-0.0, 0.0, -6.14725418],
            ],
            "deformation": [
                [1.0099504938362078, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "uuid": "013d1100-f5b7-493b-b4ac-894c85733c7e",
            "job_dir": "d",
        },
        {
            "stress": [
                [-0.21994363, 0.0, 0.0],
                [0.0, -0.1846297, 14.80836455],
                [0.0, 14.80836455, 0.40782339],
            ],
            "deformation": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, -0.02],
                [0.0, 0.0, 0.999799979995999],
            ],
            "uuid": "ab2857a6-188b-49a5-a90f-adfc30f884a7",
            "job_dir": "e",
        },
        {
            "stress": [
                [-0.17602242, 0.0, 0.0],
                [0.0, -0.16580315, 7.40412018],
                [0.0, 7.40412018, -0.01771334],
            ],
            "deformation": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, -0.01],
                [0.0, 0.0, 0.9999499987499375],
            ],
            "uuid": "6cee0242-2ff6-4c02-afe8-9c0e8c0e37b7",
            "job_dir": "f",
        },
    ]

    job = fit_elastic_tensor(conventional, deformation_data)

    response = run_locally(job, ensure_success=True)

    elastic_out = response[job.uuid][1].output
    assert elastic_out.fitting_data.failed_uuids == []
    assert elastic_out.warnings is None
    assert len(set(elastic_out.fitting_data.uuids)) == 6

    # test failure
    # remove one of the outputs
    deformation_data[0]["stress"] = None
    job = fit_elastic_tensor(conventional, deformation_data, max_failed_deformations=2)

    response = run_locally(job, ensure_success=True)

    elastic_out = response[job.uuid][1].output
    assert elastic_out.fitting_data.failed_uuids == [deformation_data[0]["uuid"]]
    assert elastic_out.warnings == [ElasticWarnings.FAILED_PERTURBATIONS.value]
    assert len(set(elastic_out.fitting_data.uuids)) == 5

    job = fit_elastic_tensor(conventional, deformation_data, max_failed_deformations=0)

    response = run_locally(job, ensure_success=False)

    assert job.uuid not in response
    assert "1 deformation calculations have failed, maximum allowed: 0" in caplog.text

    caplog.clear()
    job = fit_elastic_tensor(
        conventional, deformation_data, max_failed_deformations=0.01
    )

    response = run_locally(job, ensure_success=False)

    assert job.uuid not in response
    assert (
        "666666 fraction of deformation calculations have failed, "
        "maximum fraction allowed: 0.01" in caplog.text
    )
