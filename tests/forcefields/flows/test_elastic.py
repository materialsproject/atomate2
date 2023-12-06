from jobflow import run_locally
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.common.schemas.elastic import ElasticDocument
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import M3GNetRelaxMaker


def test_elastic_wf_with_m3gnet(clean_dir, si_structure):
    si_prim = SpacegroupAnalyzer(si_structure).get_primitive_standard_structure()

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
    # TODO (@janosh) uncomment below asserts once no longer failing with crazy values
    # (3101805 instead of 118). started happening in v0.9.0 release of matgl. reached
    # out to Shyue Ping and his group to look into this.
    # assert_allclose(elastic_output.derived_properties.k_voigt, 118.26914, atol=1e-1)
    # assert_allclose(elastic_output.derived_properties.g_voigt, 17.32737412, atol=1e-1)
    assert elastic_output.chemsys == "Si"
