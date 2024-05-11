import pytest
from jobflow import run_locally
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.common.schemas.elastic import ElasticDocument
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import MACERelaxMaker


def test_elastic_wf_with_mace(clean_dir, si_structure, test_dir):
    si_prim = SpacegroupAnalyzer(si_structure).get_primitive_standard_structure()
    model_path = f"{test_dir}/forcefields/mace/MACE.model"
    common_kwds = dict(
        calculator_kwargs={"model": model_path, "default_dtype": "float64"},
        relax_kwargs={"fmax": 0.00001},
    )

    flow = ElasticMaker(
        bulk_relax_maker=MACERelaxMaker(**common_kwds, relax_cell=True),
        elastic_relax_maker=MACERelaxMaker(**common_kwds, relax_cell=False),
    ).make(si_prim)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    elastic_output = responses[flow[-1].uuid][1].output
    assert isinstance(elastic_output, ElasticDocument)
    assert elastic_output.derived_properties.k_voigt == pytest.approx(
        9.7005429, abs=0.01
    )
    assert elastic_output.derived_properties.g_voigt == pytest.approx(
        0.002005039, abs=0.01
    )
    assert elastic_output.chemsys == "Si"
