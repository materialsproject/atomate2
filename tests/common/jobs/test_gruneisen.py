import pytest
from jobflow import run_locally
from pymatgen.core.structure import Structure

from atomate2.common.jobs.gruneisen import (
    compute_gruneisen_param,
    shrink_expand_structure,
)


def test_shrink_expand_structure(tmp_dir, si_structure: Structure):
    job = shrink_expand_structure(si_structure, perc_vol=0.01)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    assert isinstance(responses[job.output.uuid][1].output["plus"], Structure)
    assert isinstance(responses[job.output.uuid][1].output["minus"], Structure)

    assert responses[job.output.uuid][1].output[
        "plus"
    ].volume / si_structure.volume == pytest.approx(1.01)
    assert responses[job.output.uuid][1].output[
        "minus"
    ].volume / si_structure.volume == pytest.approx(0.99)


def test_compute_gruneisen_param(tmp_dir, test_dir):
    job = compute_gruneisen_param(
        mesh=(20, 20, 20),
        code="vasp",
        phonopy_yaml_paths_dict={
            "ground": str((test_dir / "vasp/Si_gruneisen/").as_posix()),
            "plus": str((test_dir / "vasp/Si_gruneisen/").as_posix()),
            "minus": str((test_dir / "vasp/Si_gruneisen/").as_posix()),
        },
        phonon_imaginary_modes_info={"ground": False, "plus": False, "minus": False},
        kpath_scheme="seekpath",
        symprec=1e-4,
        structure=Structure.from_file(test_dir / "vasp/Si_gruneisen/POSCAR"),
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # get output document
    gp_doc = responses[job.output.uuid][1].output

    # test field entries in the output doc
    assert gp_doc.phonon_runs_has_imaginary_modes.dict() == {
        "ground": False,
        "plus": False,
        "minus": False,
    }
    assert gp_doc.derived_properties.average_gruneisen == pytest.approx(
        1.1882292157682082
    )
    assert gp_doc.derived_properties.thermal_conductivity_slack == pytest.approx(
        38.861289530152796
    )
