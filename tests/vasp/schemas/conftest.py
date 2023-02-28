import pytest


def assert_schemas_equal(test_schema, valid_schema):
    """
    Recursively test all items in valid_schema are present and equal in test_schema.

    While test_schema can be a pydantic schema or dictionary, the valid schema must
    be a (nested) dictionary. This function automatically handles accessing the
    attributes of classes in the test_schema.

    Args:
        test_schema: A pydantic schema or dictionary of the schema.
        valid_schema: A (nested) dictionary specifying the key and values that must be
            present in test_schema.
    """
    from pydantic import BaseModel

    if isinstance(valid_schema, dict):
        for key, sub_valid_schema in valid_schema.items():
            if isinstance(key, str) and hasattr(test_schema, key):
                sub_test_schema = getattr(test_schema, key)
            elif not isinstance(test_schema, BaseModel):
                sub_test_schema = test_schema[key]
            else:
                raise ValueError(f"{type(test_schema)} does not have field: {key}")
            return assert_schemas_equal(sub_test_schema, sub_valid_schema)

    elif isinstance(valid_schema, list):
        for i, sub_valid_schema in enumerate(valid_schema):
            return assert_schemas_equal(test_schema[i], sub_valid_schema)

    elif isinstance(valid_schema, float):
        assert test_schema == pytest.approx(valid_schema)
    else:
        assert test_schema == valid_schema


class SchemaTestData:
    """Dummy class to be used to contain all test data information."""


class SiOptimizeDouble(SchemaTestData):
    folder = "Si_old_double_relax"
    task_files = {
        "relax2": {
            "vasprun_file": "vasprun.xml.relax2.gz",
            "outcar_file": "OUTCAR.relax2.gz",
            "volumetric_files": ["CHGCAR.relax2.gz"],
            "contcar_file": "CONTCAR.relax2.gz",
        },
        "relax1": {
            "vasprun_file": "vasprun.xml.relax1.gz",
            "outcar_file": "OUTCAR.relax1.gz",
            "volumetric_files": ["CHGCAR.relax1.gz"],
            "contcar_file": "CONTCAR.relax1.gz",
        },
    }
    objects = {"relax2": []}
    task_doc = {
        "calcs_reversed": [
            {
                "output": {
                    "vbm": 5.6147,
                    "cbm": 6.2652,
                    "bandgap": 0.6505,
                    "is_gap_direct": False,
                    "is_metal": False,
                    "transition": "(0.000,0.000,0.000)-(0.375,0.375,0.000)",
                    "direct_gap": 2.5561,
                    "run_stats": {
                        "average_memory": 0,
                        "max_memory": 28096.0,
                        "cores": 16,
                    },
                },
                "input": {
                    "incar": {"NSW": 99},
                    "nkpoints": 29,
                    "potcar_spec": [{"titel": "PAW_PBE Si 05Jan2001"}],
                    "structure": {"volume": 40.036816205493494},
                    "is_hubbard": False,
                    "hubbards": None,
                },
            }
        ],
        "analysis": {"delta_volume": 0.8638191769757384, "max_force": 0},
        "input": {
            "structure": {"volume": 40.036816205493494},
            "potcar_spec": [{"titel": "PAW_PBE Si 05Jan2001"}],
            "parameters": {"NSW": 99},
            "is_hubbard": False,
            "hubbards": None,
        },
        "output": {
            "structure": {"volume": 40.90063538246923},
            "energy": -10.84687704,
            "bandgap": 0.6505,
        },
        "custodian": [{"job": {"settings_override": None, "suffix": ".relax1"}}],
        "included_objects": (),
    }


class SiNonSCFUniform(SchemaTestData):
    from atomate2.vasp.schemas.calculation import VaspObject

    folder = "Si_band_structure/non-scf_uniform"
    task_files = {
        "standard": {
            "vasprun_file": "vasprun.xml.gz",
            "outcar_file": "OUTCAR.gz",
            "volumetric_files": ["CHGCAR.gz"],
            "contcar_file": "CONTCAR.gz",
        }
    }
    objects = {"standard": []}
    task_doc = {
        "calcs_reversed": [
            {
                "output": {
                    "vbm": 5.6162,
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
                    "structure": {"volume": 40.88829843008916},
                    "is_hubbard": False,
                    "hubbards": None,
                },
            }
        ],
        "analysis": {"delta_volume": 0, "max_force": 0.5350159115036506},
        "input": {
            "structure": {"volume": 40.88829843008916},
            "potcar_spec": [{"titel": "PAW_PBE Si 05Jan2001"}],
            "parameters": {"NSW": 0},
            "is_hubbard": False,
            "hubbards": None,
        },
        "output": {
            "structure": {"volume": 40.88829843008916},
            "energy": -10.85064059,
            "bandgap": 0.6103,
        },
        "custodian": [{"job": {"settings_override": None, "suffix": ""}}],
        "included_objects": (VaspObject.DOS, VaspObject.BANDSTRUCTURE),
    }


class SiStatic(SchemaTestData):
    from atomate2.vasp.schemas.calculation import VaspObject

    folder = "Si_band_structure/static"
    task_files = {
        "standard": {
            "vasprun_file": "vasprun.xml.gz",
            "outcar_file": "OUTCAR.gz",
            "volumetric_files": ["CHGCAR.gz"],
            "contcar_file": "CONTCAR.gz",
        }
    }
    objects = {"standard": []}
    task_doc = {
        "calcs_reversed": [
            {
                "output": {
                    "vbm": 5.6163,
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
                    "incar": {"NSW": 1},
                    "nkpoints": 29,
                    "potcar_spec": [{"titel": "PAW_PBE Si 05Jan2001"}],
                    "structure": {"volume": 40.88829843008916},
                },
            }
        ],
        "analysis": {"delta_volume": 0, "max_force": 0.0},
        "input": {
            "structure": {"volume": 40.88829843008916},
            "potcar_spec": [{"titel": "PAW_PBE Si 05Jan2001"}],
            "parameters": {"NSW": 0},
            "is_hubbard": False,
            "hubbards": None,
        },
        "output": {
            "structure": {"volume": 40.88829843008916},
            "energy": -10.84678256,
            "bandgap": 0.6506,
            "dos_properties": {
                "Si": {
                    "s": {
                        "filling": 0.624669545020562,
                        "center": -2.5151284433409815,
                        "bandwidth": 7.338662205126851,
                        "skewness": 0.6261990748648925,
                        "kurtosis": 2.0074877073276904,
                        "upper_edge": -8.105469079999999,
                    },
                    "p": {
                        "filling": 0.3911927710592045,
                        "center": 3.339269798287516,
                        "bandwidth": 5.999449671419663,
                        "skewness": 0.0173776678056677,
                        "kurtosis": 1.907790411890831,
                        "upper_edge": -0.7536690799999999,
                    },
                }
            },
        },
        "custodian": [{"job": {"settings_override": None, "suffix": ""}}],
        "included_objects": (),
    }


objects = {cls.__name__: cls for cls in SchemaTestData.__subclasses__()}


def get_test_object(object_name):
    """Get the schema test data object from the class name."""
    return objects[object_name]
