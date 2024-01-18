import numpy as np
import pytest
from emmet.core.tasks import TaskDoc
from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.common.jobs.eos import postprocess_EOS
from atomate2.vasp.jobs.eos import (
    MPGGAEosRelaxMaker,
    MPGGAEosStaticMaker,
)

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
    energies = taylor(
        volumes, EOS_pars["e0"], EOS_pars["b0"], EOS_pars["b1"], EOS_pars["v0"]
    )
    EV_dict = {
        "relax": {
            "E0": EOS_pars["e0"],
            "V0": EOS_pars["v0"],
            "energies": list(energies),
            "volumes": list(volumes),
        }
    }

    analysis_job = postprocess_EOS(EV_dict, {})
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

    # Testing that percent errors are less than 5%
    # Makes sense for testing EOS close to minimum, where Taylor series applies
    for EOS in job_output["relax"]["EOS"]:
        assert all(
            100.0 * (job_output["relax"]["EOS"][EOS][par] / EOS_pars[par] - 1.0) < 5.0
            for par in EOS_pars
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
        f"{vasp_test_dir}/{ref_paths['EOS MP GGA relax']}" "/inputs/POSCAR"
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
