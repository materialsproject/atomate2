from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from emmet.core.tasks import TaskDoc
from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.common.jobs.eos import postprocess_eos
from atomate2.vasp.jobs.eos import MPGGAEosRelaxMaker, MPGGAEosStaticMaker

if TYPE_CHECKING:
    from collections.abc import Sequence

expected_incar_relax = {
    "ISIF": 3,
    "IBRION": 2,
    "EDIFF": 1.0e-6,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LMAXMIX": 6,
    "KSPACING": 0.22,
}

expected_incar_static = {**expected_incar_relax, "NSW": 0, "IBRION": -1, "ISMEAR": -5}
expected_incar_static.pop("ISIF")


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

    analysis_job = postprocess_eos(e_v_dict, {})
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
            (job_output["relax"]["EOS"][eos][par] / eos_pars[par] - 1.0) < 0.05
            for par in eos_pars
        )


def test_mp_gga_eos_relax_maker(mock_vasp, clean_dir, vasp_test_dir):
    # map from job name to directory containing reference output files
    ref_paths = {
        "EOS MP GGA relax": "Si_EOS_MP_GGA/mp-149-PBE-EOS_MP_GGA_relax_1",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        key: {"incar_settings": list(expected_incar_relax)} for key in ref_paths
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    structure = Structure.from_file(
        f"{vasp_test_dir}/{ref_paths['EOS MP GGA relax']}/inputs/POSCAR"
    )
    maker = MPGGAEosRelaxMaker()
    job = maker.make(structure)

    # ensure flow runs successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate output
    task_doc = responses[job.uuid][1].output
    assert isinstance(task_doc, TaskDoc)
    assert task_doc.output.energy == pytest.approx(-10.849349)


def test_mp_gga_eos_static_maker(mock_vasp, clean_dir, vasp_test_dir):
    # map from job name to directory containing reference output files
    ref_paths = {
        "EOS MP GGA static": "Si_EOS_MP_GGA/mp-149-PBE-EOS_Static_0",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        key: {"incar_settings": list(expected_incar_static)} for key in ref_paths
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    input_structure = Structure.from_file(
        f"{vasp_test_dir}/Si_EOS_MP_GGA/"
        f"mp-149-PBE-EOS_Deformation_Relax_0/outputs/POSCAR.gz"
    )
    structure = Structure.from_file(
        f"{vasp_test_dir}/Si_EOS_MP_GGA/mp-149-PBE-EOS_Static_0/inputs/POSCAR"
    )
    assert input_structure == structure

    job = MPGGAEosStaticMaker().make(structure)

    # ensure flow runs successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate output
    task_doc = responses[job.uuid][1].output
    assert isinstance(task_doc, TaskDoc)
    assert task_doc.output.energy == pytest.approx(-10.547764)
