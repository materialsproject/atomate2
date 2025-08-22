import pytest
from jobflow import run_locally
from monty.serialization import loadfn

from atomate2.forcefields.flows.eos import ForceFieldEosMaker
from atomate2.utils.testing import get_job_uuid_name_map


@pytest.mark.parametrize("mlff", ["CHGNet", "MACE"])
def test_ml_ff_eos_makers(mlff: str, si_structure, clean_dir, test_dir):
    job = ForceFieldEosMaker.from_force_field_name(mlff).make(si_structure)
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
