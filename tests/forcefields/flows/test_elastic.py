import pytest
from jobflow import run_locally
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.common.schemas.elastic import ElasticDocument
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker


@pytest.mark.parametrize("convenience_constructor", [True, False])
def test_elastic_wf_with_mace(
    clean_dir, si_structure, test_dir, convenience_constructor: bool
):
    pytest.importorskip("mace")

    si_prim = SpacegroupAnalyzer(si_structure).get_primitive_standard_structure()
    model_path = f"{test_dir}/forcefields/mace/MACE.model"
    common_kwds = {
        "force_field_name": "MACE",
        "calculator_kwargs": {"model": model_path, "default_dtype": "float64"},
        "relax_kwargs": {"fmax": 0.00001},
    }

    if convenience_constructor:
        common_kwds.pop("force_field_name")
        flow = ElasticMaker.from_force_field_name(
            force_field_name="MACE",
            mlff_kwargs=common_kwds,
        ).make(si_prim)
    else:
        flow = ElasticMaker(
            bulk_relax_maker=ForceFieldRelaxMaker(**common_kwds, relax_cell=True),
            elastic_relax_maker=ForceFieldRelaxMaker(**common_kwds, relax_cell=False),
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


def test_ext_load_elastic_initialization():
    pytest.importorskip("mace")
    calculator_meta = {
        "@module": "mace.calculators",
        "@callable": "mace_mp",
    }
    maker = ElasticMaker.from_force_field_name(
        force_field_name=calculator_meta,
    )
    assert maker.bulk_relax_maker.ase_calculator_name == "mace_mp"
    assert maker.elastic_relax_maker.ase_calculator_name == "mace_mp"
