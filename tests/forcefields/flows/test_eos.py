import pytest
from jobflow import run_locally
from monty.serialization import loadfn

from atomate2.forcefields.flows.eos import ForceFieldEosMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from atomate2.utils.testing import get_job_uuid_name_map


@pytest.mark.parametrize("mlff", ["CHGNet", "MACE"])
def test_ml_ff_eos_makers(mlff: str, si_structure, clean_dir, test_dir):
    maker = ForceFieldEosMaker.from_force_field_name(mlff)
    job = maker.make(si_structure)
    for attr in ("initial_relax_maker", "eos_relax_maker"):
        assert mlff in getattr(maker, attr).force_field_name

    job_to_uuid = {v: k for k, v in get_job_uuid_name_map(job).items()}
    post_process_uuid = job_to_uuid[f"{mlff} EOS Maker postprocessing"]
    response = run_locally(job, ensure_success=True)
    output = response[post_process_uuid][1].output

    ref_data = loadfn(f"{test_dir}/forcefields/eos/{mlff}_Si_eos.json.gz")

    for key in ref_data["relax"]:
        if isinstance(key, float):
            assert output["relax"][key] == pytest.approx(ref_data["relax"][key])
        elif isinstance(key, list):
            assert all(
                output["relax"][key][idx] == pytest.approx(value)
                for idx, value in ref_data["relax"][key].items()
            )

    assert (
        ForceFieldEosMaker.from_force_field_name(
            mlff, relax_initial_structure=False
        ).initial_relax_maker
        is None
    )


def test_ext_load_eos_initialization():
    calculator_meta = {
        "@module": "mace.calculators",
        "@callable": "mace_mp",
    }
    maker = ForceFieldEosMaker.from_force_field_name(
        force_field_name=calculator_meta,
        relax_initial_structure=True,
    )
    assert isinstance(maker.initial_relax_maker, ForceFieldRelaxMaker)
    assert isinstance(maker.eos_relax_maker, ForceFieldRelaxMaker)
    assert maker.initial_relax_maker.ase_calculator_name == "mace_mp"
    assert maker.eos_relax_maker.ase_calculator_name == "mace_mp"
