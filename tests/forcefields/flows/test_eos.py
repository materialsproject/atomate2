import pytest
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
    for mlff in _mlff_to_maker:
        job = _mlff_to_maker[mlff]().make(si_structure)
        job_to_uuid = {job.name: job.uuid for job in job.jobs}
        postprocess_uuid = job_to_uuid[f"{mlff} EOS Maker_postprocess_eos"]
        response = run_locally(job)
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
