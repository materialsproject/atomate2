import numpy as np
import pytest
from jobflow import run_locally

from atomate2.common.schemas.dielectric import DielectricDocument
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.flows.dielectric import RelaxDielectricMaker
from atomate2.vasp.jobs.core import TightRelaxMaker


@pytest.mark.parametrize("use_emmet_models", [False, True])
def test_relax_dielectric(
    mock_vasp, clean_dir, memory_jobstore, si_structure, use_emmet_models
):
    # mapping from job name to directory containing test files
    ref_paths = {
        "relax 1": "Si_double_relax/relax_1",
        "relax 2": "Si_double_relax/relax_2",
        "dielectric": "Si_dielectric",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
        "dielectric": {"incar_settings": ["IBRION", "NSW", "LEPSILON"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate flow
    flow = RelaxDielectricMaker(relax_maker=DoubleRelaxMaker()).make(si_structure)
    flow.jobs[1].maker.input_set_generator.user_incar_settings["KSPACING"] = 0.5
    flow.update_maker_kwargs(
        {"_set": {"task_document_kwargs->use_emmet_models": use_emmet_models}},
        dict_mod=True,
    )

    # run the flow and ensure that it finished running successfully
    run_locally(flow, store=memory_jobstore, create_folders=True, ensure_success=True)

    # validate the output as loaded back from the job store
    output = flow.output.resolve(memory_jobstore)
    assert isinstance(output, DielectricDocument)
    np.testing.assert_allclose(
        output.epsilon_electronic, 11.4154 * np.eye(3), atol=0.01
    )
    np.testing.assert_allclose(output.epsilon_ionic, np.zeros((3, 3)), atol=0.01)
    np.testing.assert_allclose(output.epsilon_total, output.epsilon_electronic)
    assert output.refractive_index == pytest.approx(np.sqrt(11.4154), abs=1e-3)
    assert output.piezoelectric is None
    assert np.shape(output.born) == (2, 3, 3)

    # check the default flow uses a tight double relaxation and can skip it
    flow = RelaxDielectricMaker().make(si_structure)
    assert [job.name for job in flow.jobs] == [
        "double relax",
        "dielectric",
        "dielectric_analysis",
    ]
    assert isinstance(flow.jobs[0].jobs[0].maker, TightRelaxMaker)

    flow = RelaxDielectricMaker(relax_maker=None).make(si_structure)
    assert [job.name for job in flow.jobs] == ["dielectric", "dielectric_analysis"]
