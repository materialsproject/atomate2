import numpy as np
from jobflow import run_locally

from atomate2.vasp.jobs.EOS import (
    postprocess_EOS,
)


def taylor(v, E0, B0, B1, V0):
    """
    Rigorously, the EOS allows one to express E(V) as a Taylor series
    E(V) = E0 + B0 V0/2 (V/V0 - 1)**2 - (1 + B1) B0 V0/6 (V/V0 - 1)**3 + ...
    """
    return (
        E0
        + B0 * V0 * (v / V0 - 1.0) ** 2 / 2.0
        - (1.0 + B1) * B0 * V0 * (v / V0 - 1.0) ** 3 / 6.0
    )


def test_postprocess_EOS(clean_dir):
    # random params that are not unreasonable
    EOS_pars = {"e0": -1.25e2, "b0": 85.0, "b1": 4.46, "v0": 11.15}

    volumes = EOS_pars["v0"] * np.linspace(0.95, 1.05, 11)
    E = taylor(volumes, EOS_pars["e0"], EOS_pars["b0"], EOS_pars["b1"], EOS_pars["v0"])
    EV_dict = {"relax": np.transpose((volumes, E))}

    analysis_job = postprocess_EOS(EV_dict)
    response = run_locally(analysis_job, create_folders=False, ensure_success=True)
    job_output = response[analysis_job.uuid][1].output
    assert set(job_output) == {"EV", "EOS"}
    assert set(job_output["EOS"]) == {"relax"}
    for EOS in job_output["EOS"]["relax"]:
        assert all(
            100.0 * (job_output["EOS"]["relax"][EOS][par] / EOS_pars[par] - 1.0) < 5.0
            for par in EOS_pars
        )
