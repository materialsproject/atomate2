import pytest

from tests.vasp.schemas.conftest import assert_schemas_equal, get_test_object


@pytest.mark.parametrize(
    "object_name",
    [
        pytest.param("SiOptimizeDouble", id="SiOptimizeDouble"),
        pytest.param("SiStatic", id="SiStatic"),
        pytest.param("SiNonSCFUniform", id="SiNonSCFUniform"),
    ],
)
def test_analysis_summary(vasp_test_dir, object_name):
    from monty.json import MontyDecoder, jsanitize

    from atomate2.vasp.schemas.calculation import Calculation
    from atomate2.vasp.schemas.task import AnalysisSummary

    test_object = get_test_object(object_name)
    dir_name = vasp_test_dir / test_object.folder / "outputs"

    calcs_docs = []
    for task_name, files in test_object.task_files.items():
        doc, _ = Calculation.from_vasp_files(dir_name, task_name, **files)
        calcs_docs.append(doc)

    test_doc = AnalysisSummary.from_vasp_calc_docs(calcs_docs)
    valid_doc = test_object.task_doc["analysis"]
    assert_schemas_equal(test_doc, valid_doc)

    # test document can be jsanitized
    d = jsanitize(test_doc, strict=True, enum_values=True)

    # and decoded
    MontyDecoder().process_decoded(d)


@pytest.mark.parametrize(
    "object_name,task_name",
    [
        pytest.param("SiOptimizeDouble", "relax1", id="SiOptimizeDouble"),
        pytest.param("SiStatic", "standard", id="SiStatic"),
        pytest.param("SiNonSCFUniform", "standard", id="SiNonSCFUniform"),
    ],
)
def test_input_summary(vasp_test_dir, object_name, task_name):
    from monty.json import MontyDecoder, jsanitize

    from atomate2.vasp.schemas.calculation import Calculation
    from atomate2.vasp.schemas.task import InputSummary

    test_object = get_test_object(object_name)
    dir_name = vasp_test_dir / test_object.folder / "outputs"

    files = test_object.task_files[task_name]
    calc_doc, _ = Calculation.from_vasp_files(dir_name, task_name, **files)

    test_doc = InputSummary.from_vasp_calc_doc(calc_doc)
    valid_doc = test_object.task_doc["input"]
    assert_schemas_equal(test_doc, valid_doc)

    # test document can be jsanitized
    d = jsanitize(test_doc, strict=True, enum_values=True)

    # and decoded
    MontyDecoder().process_decoded(d)


@pytest.mark.parametrize(
    "object_name,task_name",
    [
        pytest.param("SiOptimizeDouble", "relax2", id="SiOptimizeDouble"),
        pytest.param("SiStatic", "standard", id="SiStatic"),
        pytest.param("SiNonSCFUniform", "standard", id="SiNonSCFUniform"),
    ],
)
def test_output_summary(vasp_test_dir, object_name, task_name):
    from monty.json import MontyDecoder, jsanitize

    from atomate2.vasp.schemas.calculation import Calculation
    from atomate2.vasp.schemas.task import OutputSummary

    test_object = get_test_object(object_name)
    dir_name = vasp_test_dir / test_object.folder / "outputs"

    files = test_object.task_files[task_name]
    calc_doc, _ = Calculation.from_vasp_files(dir_name, task_name, **files)

    test_doc = OutputSummary.from_vasp_calc_doc(calc_doc)
    valid_doc = test_object.task_doc["output"]
    assert_schemas_equal(test_doc, valid_doc)

    # test document can be jsanitized
    d = jsanitize(test_doc, strict=True, enum_values=True)

    # and decoded
    MontyDecoder().process_decoded(d)


@pytest.mark.parametrize(
    "object_name",
    [
        pytest.param("SiOptimizeDouble", id="SiOptimizeDouble"),
        pytest.param("SiStatic", id="SiStatic"),
        pytest.param("SiNonSCFUniform", id="SiNonSCFUniform"),
    ],
)
def test_task_doc(vasp_test_dir, object_name):
    from monty.json import MontyDecoder, jsanitize

    from atomate2.vasp.schemas.task import TaskDocument

    test_object = get_test_object(object_name)
    dir_name = vasp_test_dir / test_object.folder / "outputs"
    test_doc = TaskDocument.from_directory(dir_name)
    assert_schemas_equal(test_doc, test_object.task_doc)

    # test document can be jsanitized
    d = jsanitize(test_doc, strict=True, enum_values=True)

    # and decoded
    MontyDecoder().process_decoded(d)

    # Test that additional_fields works
    # test_doc = TaskDocument.from_directory(dir_name, additional_fields={"foo": "bar"})
    # assert test_doc.dict()["additional_fields"] == {"foo": "bar"}
