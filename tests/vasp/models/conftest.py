import pytest
from pydantic import BaseModel

from atomate2.vasp.schemas.calculation import VaspObject


def assert_models_equal(test_model, valid_model):
    """
    Recursively test that all items in valid_model are present and equal in test_model.

    While test_model can be a pydantic model or dictionary, the valid model must
    be a (nested) dictionary. This function automatically handles accessing the
    attributes of classes in the test_model.

    Args:
        test_model: A pydantic model or dictionary of the model.
        valid_model: A (nested) dictionary specifying the key and values that must be
            present in test_model.
    """
    if isinstance(valid_model, dict):
        for key, sub_valid_model in valid_model.items():
            if isinstance(key, str) and hasattr(test_model, key):
                sub_test_model = getattr(test_model, key)
            elif not isinstance(test_model, BaseModel):
                sub_test_model = test_model[key]
            else:
                raise ValueError(
                    "{} does not have field: {}".format(type(test_model), key)
                )
            return assert_models_equal(sub_test_model, sub_valid_model)

    elif isinstance(valid_model, list):
        for i, sub_valid_model in enumerate(valid_model):
            return assert_models_equal(test_model[i], sub_valid_model)

    elif isinstance(valid_model, float):
        assert test_model == pytest.approx(valid_model)
    else:
        assert test_model == valid_model


class ModelTestData:
    """Dummy class to be used to contain all test data information."""


class SiOptimizeDouble(ModelTestData):
    folder = "Si_structure_optimization_double"
    task_files = {
        "relax1": {
            "vasprun_file": "vasprun.xml.relax1.gz",
            "outcar_file": "OUTCAR.relax1.gz",
            "volumetric_files": ["CHGCAR.relax1.gz"],
        },
        "relax2": {
            "vasprun_file": "vasprun.xml.relax2.gz",
            "outcar_file": "OUTCAR.relax2.gz",
            "volumetric_files": ["CHGCAR.relax2.gz"],
        },
    }
    objects = {"relax1": []}
    task_doc = {
        "calcs_reversed": [
            {
                "output": {
                    "vbm": 5.6149,
                    "cbm": 6.2649,
                    "bandgap": 0.65,
                    "is_gap_direct": False,
                    "is_metal": False,
                    "transition": "(0.000,0.000,0.000)-(0.375,0.375,0.000)",
                    "direct_gap": 2.5561,
                    "run_stats": {
                        "average_memory": 0,
                        "max_memory": 28640.0,
                        "cores": 16,
                    },
                },
                "input": {
                    "incar": {"NSW": 99},
                    "nkpoints": 29,
                    "potcar_spec": [{"titel": "PAW_PBE Si 05Jan2001"}],
                    "structure": {"volume": 40.036816205493494},
                },
            }
        ],
        "analysis": {"delta_volume": 0.8638191769757384, "max_force": 0},
        "input": {
            "structure": {"volume": 40.036816205493494},
            "potcar_spec": [{"titel": "PAW_PBE Si 05Jan2001"}],
            "parameters": {"NSW": 99},
        },
        "output": {
            "structure": {"volume": 40.90063538246923},
            "energy": -10.84687704,
            "bandgap": 0.6505,
        },
        "custodian": [{"job": {"settings_override": None, "suffix": ".relax1"}}],
        "included_objects": (),
    }


class SiNonSCFUniform(ModelTestData):
    folder = "Si_nscf_uniform"
    task_files = {
        "standard": {
            "vasprun_file": "vasprun.xml.gz",
            "outcar_file": "OUTCAR.gz",
            "volumetric_files": ["CHGCAR.gz"],
        }
    }
    objects = {"standard": [VaspObject.DOS, VaspObject.BANDSTRUCTURE]}
    task_doc = {
        "calcs_reversed": [
            {
                "output": {
                    "vbm": 5.614,
                    "cbm": 6.2243,
                    "bandgap": 0.6103,
                    "is_gap_direct": False,
                    "is_metal": False,
                    "transition": "(0.000,0.000,0.000)-(0.000,0.421,0.000)",
                    "direct_gap": 2.5563,
                    "run_stats": {
                        "average_memory": 0,
                        "max_memory": 31004.0,
                        "cores": 16,
                    },
                },
                "input": {
                    "incar": {"NSW": 0},
                    "nkpoints": 220,
                    "potcar_spec": [{"titel": "PAW_PBE Si 05Jan2001"}],
                    "structure": {"volume": 40.90250280144601},
                },
            }
        ],
        "analysis": {"delta_volume": 0, "max_force": 0.5350159115036506},
        "input": {
            "structure": {"volume": 40.90250280144601},
            "potcar_spec": [{"titel": "PAW_PBE Si 05Jan2001"}],
            "parameters": {"NSW": 0},
        },
        "output": {
            "structure": {"volume": 40.90250280144601},
            "energy": -10.85064059,
            "bandgap": 0.6103,
        },
        "custodian": [{"job": {"settings_override": None, "suffix": ""}}],
        "included_objects": (VaspObject.DOS, VaspObject.BANDSTRUCTURE),
    }


class SiStatic(ModelTestData):
    folder = "Si_static"
    task_files = {
        "standard": {
            "vasprun_file": "vasprun.xml.gz",
            "outcar_file": "OUTCAR.gz",
            "volumetric_files": ["CHGCAR.gz"],
        }
    }
    objects = {"standard": [VaspObject.DOS, VaspObject.BANDSTRUCTURE]}
    task_doc = {
        "calcs_reversed": [
            {
                "output": {
                    "vbm": 5.6138,
                    "cbm": 6.2644,
                    "bandgap": 0.6506,
                    "is_gap_direct": False,
                    "is_metal": False,
                    "transition": "(0.000,0.000,0.000)-(0.000,0.375,0.000)",
                    "direct_gap": 2.5563,
                    "run_stats": {
                        "average_memory": 0,
                        "max_memory": 28124.0,
                        "cores": 16,
                    },
                },
                "input": {
                    "incar": {"NSW": 0},
                    "nkpoints": 29,
                    "potcar_spec": [{"titel": "PAW_PBE Si 05Jan2001"}],
                    "structure": {"volume": 40.90250280144601},
                },
            }
        ],
        "analysis": {"delta_volume": 0, "max_force": 0.0},
        "input": {
            "structure": {"volume": 40.90250280144601},
            "potcar_spec": [{"titel": "PAW_PBE Si 05Jan2001"}],
            "parameters": {"NSW": 0},
        },
        "output": {
            "structure": {"volume": 40.90250280144601},
            "energy": -10.84678256,
            "bandgap": 0.6506,
        },
        "custodian": [{"job": {"settings_override": None, "suffix": ""}}],
        "included_objects": (VaspObject.DOS, VaspObject.BANDSTRUCTURE),
    }


objects = {cls.__name__: cls for cls in ModelTestData.__subclasses__()}


def get_test_object(object_name):
    """Get the model test data object from the class name."""
    return objects[object_name]
