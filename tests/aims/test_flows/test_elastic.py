import os

import pytest
from jobflow import run_locally
from numpy.testing import assert_allclose
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.aims.flows.elastic import ElasticMaker
from atomate2.aims.jobs.core import RelaxMaker
from atomate2.common.schemas.elastic import ElasticDocument

cwd = os.getcwd()


@pytest.mark.parametrize("conventional", [False, True])
def test_elastic(si, tmp_path, mock_aims, species_dir, conventional):
    ref_paths = {
        "Relaxation calculation (fixed cell) 1/6": "elastic-si-rel-1",
        "Relaxation calculation (fixed cell) 2/6": "elastic-si-rel-2",
        "Relaxation calculation (fixed cell) 3/6": "elastic-si-rel-3",
        "Relaxation calculation (fixed cell) 4/6": "elastic-si-rel-4",
        "Relaxation calculation (fixed cell) 5/6": "elastic-si-rel-5",
        "Relaxation calculation (fixed cell) 6/6": "elastic-si-rel-6",
        "Relaxation calculation": "elastic-si-bulk-relax",
    }

    # settings passed to fake_run_aims; adjust these to check for certain input settings
    fake_run_aims_kwargs = {}

    # automatically use fake FHI-aims
    mock_aims(ref_paths, fake_run_aims_kwargs)

    parameters = {
        "k_grid": [2, 2, 2],
        "species_dir": (species_dir / "light").as_posix(),
    }

    # generate flow
    si_sga = SpacegroupAnalyzer(si).get_conventional_standard_structure()
    maker = ElasticMaker(
        bulk_relax_maker=RelaxMaker.full_relaxation(
            user_params=dict(**parameters, rlsy_symmetry="all")
        ),
        elastic_relax_maker=RelaxMaker.fixed_cell_relaxation(
            user_params=dict(
                **parameters, compute_analytical_stress=True, rlsy_symmetry=None
            )
        ),
    )
    flow = maker.make(si_sga, conventional=conventional)

    # Run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    os.chdir(cwd)

    # validation on the outputs
    elastic_output = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(elastic_output, ElasticDocument)

    assert_allclose(
        elastic_output.elastic_tensor.ieee_format,
        [
            [147.279167, 56.2746603, 56.2746603, 0.0, 0.0, 0.0],
            [56.2746603, 147.279167, 56.2746603, 0.0, 0.0, 0.0],
            [56.2746603, 56.2746603, 147.279167, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 75.9240547, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 75.9240547, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 75.9240547],
        ],
        atol=1e-6,
    )
    assert elastic_output.chemsys == "Si"
