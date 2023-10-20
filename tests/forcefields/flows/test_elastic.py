import numpy as np

from atomate2.common.schemas.elastic import ElasticDocument
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import M3GNetRelaxMaker


def test_elastic_wf(clean_dir, si_structure):
    from jobflow import run_locally
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    # TODO test with Alumnium
    si_prim = SpacegroupAnalyzer(si_structure).get_primitive_standard_structure()

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
    # !!! validation on the outputs
    assert isinstance(elastic_output, ElasticDocument)
    assert np.allclose(
        elastic_output.elastic_tensor.ieee_format,
        [
            [
                137.8521169761535,
                108.47766069253291,
                108.47765834315278,
                8.381512878444252e-07,
                1.7168846136878433e-06,
                -7.857832305266632e-08,
            ],
            [
                108.47766069253291,
                137.85210595521275,
                108.47765400689046,
                1.1331095025033052e-06,
                1.2699645849546667e-06,
                -7.38629015073859e-08,
            ],
            [
                108.47765834315277,
                108.47765400689046,
                137.85209998408476,
                1.0651125485225092e-06,
                1.6138557475735372e-06,
                -5.812369670874092e-08,
            ],
            [
                8.381512878652419e-07,
                1.133109502524122e-06,
                1.0651125484947536e-06,
                19.08747291002524,
                -1.0227298701797934e-08,
                2.2345998151857803e-07,
            ],
            [
                1.716884613678302e-06,
                1.2699645849477278e-06,
                1.6138557475642857e-06,
                -1.0227298701797246e-08,
                19.087473673023766,
                1.4747912892211538e-07,
            ],
            [
                -7.857832305729226e-08,
                -7.386290150738587e-08,
                -5.812369671105389e-08,
                2.2345998151857816e-07,
                1.474791289221158e-07,
                19.087474086415153,
            ],
        ],
        atol=1e-3,
    )

    assert elastic_output.chemsys == "Si"
