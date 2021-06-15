import pytest

from tests.vasp.schemas.conftest import assert_schemas_equal, get_test_object


def test_init():
    from atomate2.vasp.schemas.calculation import (
        Calculation,
        CalculationInput,
        CalculationOutput,
        RunStatistics,
    )

    c = CalculationInput()
    assert c is not None

    c = CalculationOutput()
    assert c is not None

    c = RunStatistics()
    assert c is not None

    c = Calculation()
    assert c is not None


@pytest.mark.parametrize(
    "object_name,task_name",
    [
        pytest.param("SiOptimizeDouble", "relax1", id="SiOptimizeDouble"),
        pytest.param("SiStatic", "standard", id="SiStatic"),
        pytest.param("SiNonSCFUniform", "standard", id="SiNonSCFUniform"),
    ],
)
def test_calculation_input(vasp_test_dir, object_name, task_name):
    from monty.json import jsanitize
    from pymatgen.io.vasp import Vasprun

    from atomate2.vasp.schemas.calculation import CalculationInput

    test_object = get_test_object(object_name)
    vasprun_file = vasp_test_dir / test_object.folder / "outputs"
    vasprun_file /= test_object.task_files[task_name]["vasprun_file"]
    test_doc = CalculationInput.from_vasprun(Vasprun(vasprun_file))
    valid_doc = test_object.task_doc["calcs_reversed"][0]["input"]
    assert_schemas_equal(test_doc, valid_doc)

    # test document can be jsanitized
    jsanitize(test_doc, strict=True, allow_bson=True)


@pytest.mark.parametrize(
    "object_name,task_name",
    [
        pytest.param("SiOptimizeDouble", "relax1", id="SiOptimizeDouble"),
        pytest.param("SiStatic", "standard", id="SiStatic"),
        pytest.param("SiNonSCFUniform", "standard", id="SiNonSCFUniform"),
    ],
)
def test_calculation_output(vasp_test_dir, object_name, task_name):
    from monty.json import jsanitize
    from pymatgen.io.vasp import Outcar, Vasprun

    from atomate2.vasp.schemas.calculation import CalculationOutput

    test_object = get_test_object(object_name)
    folder = vasp_test_dir / test_object.folder / "outputs"
    vasprun_file = folder / test_object.task_files[task_name]["vasprun_file"]
    outcar_file = folder / test_object.task_files[task_name]["outcar_file"]
    vasprun = Vasprun(vasprun_file)
    outcar = Outcar(outcar_file)
    test_doc = CalculationOutput.from_vasp_outputs(vasprun, outcar)
    valid_doc = test_object.task_doc["calcs_reversed"][0]["output"]
    assert_schemas_equal(test_doc, valid_doc)

    # test document can be jsanitized
    jsanitize(test_doc, strict=True, allow_bson=True)


@pytest.mark.parametrize(
    "object_name,task_name",
    [
        pytest.param("SiOptimizeDouble", "relax1", id="SiOptimizeDouble"),
        pytest.param("SiStatic", "standard", id="SiStatic"),
        pytest.param("SiNonSCFUniform", "standard", id="SiNonSCFUniform"),
    ],
)
def test_run_statistics(vasp_test_dir, object_name, task_name):
    from monty.json import jsanitize
    from pymatgen.io.vasp import Outcar

    from atomate2.vasp.schemas.calculation import RunStatistics

    test_object = get_test_object(object_name)
    folder = vasp_test_dir / test_object.folder / "outputs"
    outcar_file = folder / test_object.task_files[task_name]["outcar_file"]
    outcar = Outcar(outcar_file)
    test_doc = RunStatistics.from_outcar(outcar)
    valid_doc = test_object.task_doc["calcs_reversed"][0]["output"]["run_stats"]
    assert_schemas_equal(test_doc, valid_doc)

    # test document can be jsanitized
    jsanitize(test_doc, strict=True, allow_bson=True)


@pytest.mark.parametrize(
    "object_name,task_name",
    [
        pytest.param("SiOptimizeDouble", "relax1", id="SiOptimizeDouble"),
        pytest.param("SiStatic", "standard", id="SiStatic"),
        pytest.param("SiNonSCFUniform", "standard", id="SiNonSCFUniform"),
    ],
)
def test_calculation(vasp_test_dir, object_name, task_name):
    from monty.json import jsanitize

    from atomate2.vasp.schemas.calculation import Calculation

    test_object = get_test_object(object_name)
    dir_name = vasp_test_dir / test_object.folder / "outputs"
    files = test_object.task_files[task_name]

    test_doc, objects = Calculation.from_vasp_files(dir_name, task_name, **files)
    valid_doc = test_object.task_doc["calcs_reversed"][0]
    assert_schemas_equal(test_doc, valid_doc)
    assert set(objects.keys()) == set(test_object.objects[task_name])

    # test document can be jsanitized
    jsanitize(test_doc, strict=True, allow_bson=True)
