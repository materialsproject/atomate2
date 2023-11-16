import torch
from jobflow import run_locally
from numpy.testing import assert_allclose
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.common.schemas.elastic import ElasticDocument
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import M3GNetRelaxMaker


def test_elastic_wf(clean_dir, si_structure):
    si_prim = SpacegroupAnalyzer(si_structure).get_primitive_standard_structure()

    # FIXME - brittle due to inability to adjust dtypes in M3GNetRelaxMaker
    torch.set_default_dtype(torch.float32)

    # !!! Generate job
    job = ElasticMaker(
        bulk_relax_maker=M3GNetRelaxMaker(
            relax_cell=True, relax_kwargs={"fmax": 0.00001}
        ),
        elastic_relax_maker=M3GNetRelaxMaker(
            relax_cell=False, relax_kwargs={"fmax": 0.00001}
        ),
    ).make(si_prim)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)
    elastic_output = responses[job.jobs[-1].uuid][1].output
    assert isinstance(elastic_output, ElasticDocument)
    assert_allclose(elastic_output.derived_properties.k_voigt, 118.26914, atol=1e-1)
    assert_allclose(elastic_output.derived_properties.g_voigt, 17.32737412, atol=1e-1)
    assert elastic_output.chemsys == "Si"
