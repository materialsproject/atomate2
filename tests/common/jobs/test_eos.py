from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jobflow import run_locally
from pytest import approx

from atomate2.common.jobs.eos import apply_strain_to_structure, postprocess_eos

if TYPE_CHECKING:
    from collections.abc import Sequence


def taylor(
    v: float | Sequence, e0: float, b0: float, b1: float, v0: float
) -> float | Sequence:
    """
    Taylor series expansion of an equation of state about the E(V) minimum.

    Parameters
    ----------
    v : float | Sequence
        A volume or list of volumes to evaluate the EOS at
    e0 : float
        The EOS minimum energy, i.e., E(V0), where V0 is the
        equilibrium volume
    b0 : float
        The bulk modulus at V0
    b1 : float
        The pressure derivative of the bulk modulus at V0
    v0: float
        The equilibrium volume

    Rigorously, the EOS allows one to express E(V) as a Taylor series
    E(V) = E0 + B0 V0/2 (V/V0 - 1)**2 - (1 + B1) B0 V0/6 (V/V0 - 1)**3 + ...
    """
    return (
        e0
        + b0 * v0 * (v / v0 - 1.0) ** 2 / 2.0
        - (1.0 + b1) * b0 * v0 * (v / v0 - 1.0) ** 3 / 6.0
    )


def test_postprocess_eos(clean_dir):
    # random params that are not unreasonable
    eos_pars = {"e0": -1.25e2, "b0": 85.0, "b1": 4.46, "v0": 11.15}

    volumes = eos_pars["v0"] * np.linspace(0.95, 1.05, 11)
    energies = taylor(volumes, *[eos_pars[key] for key in ("e0", "b0", "b1", "v0")])
    e_v_dict = {
        "relax": {
            "E0": eos_pars["e0"],
            "V0": eos_pars["v0"],
            "energies": list(energies),
            "volumes": list(volumes),
        }
    }

    analysis_job = postprocess_eos(e_v_dict)
    response = run_locally(analysis_job, create_folders=False, ensure_success=True)
    job_output = response[analysis_job.uuid][1].output
    assert set(job_output["relax"]) == {"E0", "V0", "energies", "volumes", "EOS"}
    assert set(job_output["relax"]["EOS"]) == {
        "murnaghan",
        "birch",
        "birch_murnaghan",
        "pourier_tarantola",
        "vinet",
    }

    # Testing that percent errors are less than 5%. Makes sense for
    # testing EOS close to minimum, where Taylor series applies
    for eos in job_output["relax"]["EOS"]:
        assert all(
            abs(job_output["relax"]["EOS"][eos][par] / eos_pars[par] - 1.0) < 0.05
            for par in eos_pars
        )


def test_apply_strain_to_structure(clean_dir, si_structure):
    strains = [1.0 + eps for eps in (-0.05, 0.0, 0.05)]

    expected_volumes = [strain**3 * si_structure.volume for strain in strains]
    deformations = [strain * np.identity(3) for strain in strains]

    job = apply_strain_to_structure(si_structure, deformations)
    response = run_locally(job)
    transformations = response[job.uuid][1].output

    assert all(
        transformations[i].final_structure.volume == approx(expected)
        for i, expected in enumerate(expected_volumes)
    )
