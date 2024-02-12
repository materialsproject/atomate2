import pytest
import torch
from jobflow import run_locally
from monty.serialization import loadfn

from atomate2.forcefields.flows.eos import CHGNetEosMaker, M3GNetEosMaker, MACEEosMaker

_mlff_to_maker = {
    "CHGNet": CHGNetEosMaker,
    "M3GNet": M3GNetEosMaker,
    "MACE": MACEEosMaker,
}


@pytest.mark.parametrize("mlff", list(_mlff_to_maker))
def test_ml_ff_eos_makers(mlff: str, si_structure, clean_dir, test_dir):
    # MACE changes the default dtype, ensure consistent dtype here
    torch.set_default_dtype(torch.float32)

    job = _mlff_to_maker[mlff]().make(si_structure)
    job_to_uuid = {job.name: job.uuid for job in job.jobs}
    postprocess_uuid = job_to_uuid[f"{mlff} EOS Maker postprocessing"]
    response = run_locally(job, ensure_success=True)
    output = response[postprocess_uuid][1].output

    ref_data = loadfn(f"{test_dir}/forcefields/eos/{mlff}_Si_eos.json.gz")

    for key in ref_data["relax"]:
        if isinstance(key, float):
            assert output["relax"][key] == pytest.approx(ref_data["relax"][key])
        elif isinstance(key, list):
            assert all(
                output["relax"][key][i] == pytest.approx(value)
                for i, value in ref_data["relax"][key].items()
            )
