import pytest

from atomate2.vasp.models.calculation import (
    VaspCalcDoc,
    VaspInputDoc,
    VaspOutputDoc,
    RunStatistics,
)
from pymatgen.io.vasp import Vasprun, Outcar
from tests.vasp.models.conftest import assert_models_equal, get_test_object


@pytest.mark.parametrize(
    "object_name,task_name",
    [
        pytest.param("SiOptimizeDouble", "relax1", id="SiOptimizeDouble"),
        pytest.param("SiStatic", "standard", id="SiStatic"),
        pytest.param("SiNonSCFUniform", "standard", id="SiNonSCFUniform"),
    ],
)
def test_vasp_input_doc(vasp_test_dir, object_name, task_name):
    test_object = get_test_object(object_name)
    vasprun_file = vasp_test_dir / test_object.folder / "outputs"
    vasprun_file /= test_object.task_files[task_name]["vasprun_file"]
    test_doc = VaspInputDoc.from_vasprun(Vasprun(vasprun_file))
    valid_doc = test_object.task_doc["calcs_reversed"][0]["input"]
    assert_models_equal(test_doc, valid_doc)


@pytest.mark.parametrize(
    "object_name,task_name",
    [
        pytest.param("SiOptimizeDouble", "relax1", id="SiOptimizeDouble"),
        pytest.param("SiStatic", "standard", id="SiStatic"),
        pytest.param("SiNonSCFUniform", "standard", id="SiNonSCFUniform"),
    ],
)
def test_vasp_output_doc(vasp_test_dir, object_name, task_name):
    test_object = get_test_object(object_name)
    folder = vasp_test_dir / test_object.folder / "outputs"
    vasprun_file = folder / test_object.task_files[task_name]["vasprun_file"]
    outcar_file = folder / test_object.task_files[task_name]["outcar_file"]
    vasprun = Vasprun(vasprun_file)
    outcar = Outcar(outcar_file)
    test_doc = VaspOutputDoc.from_vasp_outputs(vasprun, outcar)
    valid_doc = test_object.task_doc["calcs_reversed"][0]["output"]
    assert_models_equal(test_doc, valid_doc)


@pytest.mark.parametrize(
    "object_name,task_name",
    [
        pytest.param("SiOptimizeDouble", "relax1", id="SiOptimizeDouble"),
        pytest.param("SiStatic", "standard", id="SiStatic"),
        pytest.param("SiNonSCFUniform", "standard", id="SiNonSCFUniform"),
    ],
)
def test_run_statistics(vasp_test_dir, object_name, task_name):
    test_object = get_test_object(object_name)
    folder = vasp_test_dir / test_object.folder / "outputs"
    outcar_file = folder / test_object.task_files[task_name]["outcar_file"]
    outcar = Outcar(outcar_file)
    test_doc = RunStatistics.from_outcar(outcar)
    valid_doc = test_object.task_doc["calcs_reversed"][0]["output"]["run_stats"]
    assert_models_equal(test_doc, valid_doc)


@pytest.mark.parametrize(
    "object_name,task_name",
    [
        pytest.param("SiOptimizeDouble", "relax1", id="SiOptimizeDouble"),
        pytest.param("SiStatic", "standard", id="SiStatic"),
        pytest.param("SiNonSCFUniform", "standard", id="SiNonSCFUniform"),
    ],
)
def test_vasp_calc_doc(vasp_test_dir, object_name, task_name):
    test_object = get_test_object(object_name)
    dir_name = vasp_test_dir / test_object.folder / "outputs"
    files = test_object.task_files[task_name]

    test_doc, objects = VaspCalcDoc.from_vasp_files(dir_name, task_name, **files)
    valid_doc = test_object.task_doc["calcs_reversed"][0]
    assert_models_equal(test_doc, valid_doc)
    assert set(objects.keys()) == set(test_object.objects[task_name])
