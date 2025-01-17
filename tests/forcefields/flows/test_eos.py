import pytest
from jobflow import run_locally
from monty.serialization import loadfn

from atomate2.forcefields import MLFF
from atomate2.forcefields.flows.eos import (
    CHGNetEosMaker,
    ForceFieldEosMaker,
    # M3GNetEosMaker,
    MACEEosMaker,
)

ff_maker_map = {
    MLFF.CHGNet.value: CHGNetEosMaker,
    # skip m3gnet due M3GNet requiring DGL which is PyTorch 2.4 incompatible
    # raises "FileNotFoundError: Cannot find DGL C++ libgraphbolt_pytorch_2.4.1.so"
    # MLFF.M3GNet.value: M3GNetEosMaker,
    MLFF.MACE.value: MACEEosMaker,
}


@pytest.mark.parametrize("mlff", ff_maker_map)
def test_ml_ff_eos_makers(mlff: str, si_structure, clean_dir, test_dir):
    job = ForceFieldEosMaker.from_force_field_name(mlff).make(si_structure)
    job_to_uuid = {job.name: job.uuid for job in job.jobs}
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

    with pytest.warns(FutureWarning):
        ff_maker_map[mlff]()
