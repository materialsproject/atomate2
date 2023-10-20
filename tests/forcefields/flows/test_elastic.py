import numpy as np
from pymatgen.core.structure import Structure
from atomate2.common.schemas.elastic import ElasticDocument
from atomate2.forcefields.flows.elastic import ElasticMaker

def test_elastic_wf(clean_dir, si_structure):
    from jobflow import run_locally
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    si_prim = SpacegroupAnalyzer(si_structure).get_primitive_standard_structure()

    # !!! Generate job
    job = ElasticMaker().make(si_prim)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)
    elastic_output=responses[job.jobs[-1].uuid][1].output
    # !!! validation on the outputs
    assert isinstance(elastic_output, ElasticDocument)
    assert np.allclose(
        elastic_output.elastic_tensor.ieee_format,
        [
            [3977.9849215966215, 0.4229787220627818, 0.4229787364421082, -3.2295162581729646e-08, -7.425146345400383e-08, -1.024151023658735e-07],
            [0.4229787220627818, 3977.9849689908447, 0.4229787389604583, -7.217137169487033e-08, -3.3226000335063545e-08, -0.0004310059911418357],
            [0.42297873644210815, 0.42297873896045834, 3977.9852398738194, -0.0003037261372428852, -0.00031248037772480935, -4.580496493506366e-08],
            [-3.229516258175224e-08, -7.217137169490658e-08, -0.0003037261372428852, 0.26113455179880346, -2.8269500425138218e-08, -2.0512746010223815e-08],
            [-7.425146345395245e-08, -3.32260003350997e-08, -0.0003124803777248094, -2.8269500425137655e-08, 0.26113455024456755, -1.9938074251124073e-08],
            [-1.0241510236596086e-07, -0.00043100599114183596, -4.5804964935032035e-08, -2.051274601022437e-08, -1.9938074251124345e-08, 0.26113454137567343],
        ],
        atol=1e-3,
    )
    assert elastic_output.chemsys == "Si"


