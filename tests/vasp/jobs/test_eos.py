from __future__ import annotations

from emmet.core.tasks import TaskDoc
from jobflow import run_locally
from pymatgen.core import Structure
from pytest import approx

from atomate2.vasp.jobs.eos import MPGGAEosRelaxMaker, MPGGAEosStaticMaker

expected_incar_relax = {
    "ISIF": 3,
    "IBRION": 2,
    "EDIFF": 1e-6,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LMAXMIX": 6,
    "KSPACING": 0.22,
}

expected_incar_static = {**expected_incar_relax, "NSW": 0, "IBRION": -1, "ISMEAR": -5}
expected_incar_static.pop("ISIF")


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
    assert task_doc.output.energy == approx(-10.849349)


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
    assert task_doc.output.energy == approx(-10.547764)
