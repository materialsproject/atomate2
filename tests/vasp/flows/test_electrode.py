from __future__ import annotations

from unittest import mock

import pytest
from jobflow.settings import JobflowSettings


@pytest.fixture()
def mock_jobflow_settings(memory_jobstore):
    """Set the UID_TYPE to "ulid" to make sure the documents can be sorted.

    See: https://github.com/materialsproject/jobflow/issues/519#issuecomment-1906850096
    """

    settings = JobflowSettings(JOB_STORE=memory_jobstore, UID_TYPE="ulid")

    with mock.patch("jobflow.SETTINGS", settings):
        yield


def test_electrode_makers(mock_vasp, clean_dir, test_dir, mock_jobflow_settings):
    from emmet.core.electrode import InsertionElectrodeDoc
    from jobflow import OutputReference, run_locally
    from monty.serialization import loadfn
    from pymatgen.core import Structure

    from atomate2.vasp.flows.core import RelaxMaker, StaticMaker
    from atomate2.vasp.flows.electrode import ElectrodeInsertionMaker
    from atomate2.vasp.powerups import (
        update_user_incar_settings,
        update_user_kpoints_settings,
    )
    from atomate2.vasp.sets.mp import (
        MPMetaGGARelaxSetGenerator,
        MPMetaGGAStaticSetGenerator,
    )
    # mock the default setting

    # mapping from job name to directory containing test files
    ref_paths = {
        "relax": "H_Graphite/relax",
        "relax 0 (0)": "H_Graphite/relax_0_(0)",
        "relax 1 (0)": "H_Graphite/relax_1_(0)",
        "relax 1 (1)": "H_Graphite/relax_1_(1)",
        "relax 1 (2)": "H_Graphite/relax_1_(2)",
        "static 0": "H_Graphite/static_0",
        "static 1": "H_Graphite/static_1",
    }

    fake_run_vasp_kwargs = {
        "relax": {
            "incar_settings": ["NSW", "ISIF"],
            "check_inputs": ["incar", "poscar"],
        },
        "relax 0 (0)": {"incar_settings": ["NSW"], "check_inputs": ["incar"]},
        "relax 1 (0)": {"incar_settings": ["NSW"], "check_inputs": ["incar"]},
        "relax 1 (1)": {"incar_settings": ["NSW"], "check_inputs": ["incar"]},
        "relax 1 (2)": {"incar_settings": ["NSW"], "check_inputs": ["incar"]},
        "static 0": {"incar_settings": ["NSW"], "check_inputs": ["incar"]},
        "static 1": {"incar_settings": ["NSW"], "check_inputs": ["incar"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # create the workflow
    struct = Structure.from_file(test_dir / "vasp/H_Graphite/C4.vasp")
    h_entry = loadfn(test_dir / "vasp/H_Graphite/H_entry.json")
    single_relax_maker = RelaxMaker(input_set_generator=MPMetaGGARelaxSetGenerator())
    static_maker = StaticMaker(
        input_set_generator=MPMetaGGAStaticSetGenerator(), task_document_kwargs={}
    )

    maker = ElectrodeInsertionMaker(
        relax_maker=single_relax_maker, static_maker=static_maker
    )
    flow = maker.make(
        struct, inserted_element="H", n_steps=2, working_ion_entry=h_entry
    )

    flow = update_user_kpoints_settings(flow, {"grid_density": 88})
    flow = update_user_incar_settings(
        flow, {"NGX": 18, "NGY": 18, "NGZ": 60, "ISIF": 2, "EDIFFG": -0.1}
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        flow, create_folders=True, ensure_success=True, raise_immediately=True
    )

    inserted_formulas = []
    ie_doc = None
    for res in responses.values():
        for r in res.values():
            if not isinstance(r.output, OutputReference) and hasattr(
                r.output, "formula_pretty"
            ):
                inserted_formulas.append(
                    f"{r.output.formula_pretty}-{r.output.task_label.split()[0]}"
                )
            if isinstance(r.output, InsertionElectrodeDoc):
                ie_doc = r.output

    inserted_formulas.sort()

    # C-relax, C-static
    # HC4-relax (1x first insertion)
    # HC4-static
    # HC2-relax, (3x second insertion)
    assert inserted_formulas == [
        "C-relax",
        "C-static",
        "HC2-relax",
        "HC2-relax",
        "HC2-relax",
        "HC4-relax",
        "HC4-static",
    ]

    # None of the secondary insertions were topotactic
    assert len(ie_doc.adj_pairs) == 1
