import pytest
from emmet.core.tasks import TaskDoc
from jobflow import run_locally

from atomate2.vasp.flows.mvl import MVLGWBandStructureMaker


def test_mvl_gw(mock_vasp, clean_dir, si_structure):
    from atomate2.vasp.powerups import (
        update_user_incar_settings,
        update_user_kpoints_settings,
        update_user_potcar_functional,
    )

    # mapping from job name to directory containing test files
    ref_paths = {
        "MVL G0W0": "Si_G0W0/MVL_G0W0",
        "MVL nscf": "Si_G0W0/MVL_nscf",
        "MVL static": "Si_G0W0/MVL_static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "MVL G0W0": {"incar_settings": ["NSW", "ISMEAR"]},
        "MVL nscf": {"incar_settings": ["NSW", "ISMEAR"]},
        "MVL static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    flow = MVLGWBandStructureMaker().make(si_structure)
    flow = update_user_kpoints_settings(flow, kpoints_updates={"reciprocal_density": 5})
    flow = update_user_potcar_functional(flow, potcar_functional="PBE_54")
    flow = update_user_incar_settings(flow, incar_updates={"ISPIN": 1})

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate the outputs
    output1 = responses[flow.jobs[0].uuid][1].output
    output2 = responses[flow.jobs[1].uuid][1].output
    output3 = responses[flow.jobs[2].uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert isinstance(output2, TaskDoc)
    assert isinstance(output3, TaskDoc)
    assert output1.output.energy == pytest.approx(-10.22237938)
    assert output1.output.bandgap == pytest.approx(0.7161)
    assert output2.output.energy == pytest.approx(-10.22215331)
    assert output3.output.bandgap == pytest.approx(1.3486000000000002)
