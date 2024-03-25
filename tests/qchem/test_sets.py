import os

import pytest
from pymatgen.core.structure import Molecule
from pymatgen.io.qchem.inputs import QCInput

from atomate2.qchem.sets.base import QCInputSet
from atomate2.qchem.sets.core import SinglePointSetGenerator


@pytest.fixture(scope="module")
def water_mol() -> Molecule:
    """Dummy molecular structure for water as a test molecule."""
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.121], [-0.783, 0.0, -0.485], [0.783, 0.0, -0.485]],
    )


@pytest.mark.parametrize(
    "molecule,overwrite_inputs",
    [
        ("water_mol", {"rem": {"ideriv": 1, "method": "B97-D3", "dft_d": "D3_BJ"}}),
        ("water_mol", {"opt": {"CONSTRAINT": ["stre 1 2 0.96"]}}),
        (
            "water_mol",
            {
                "rem": {"solvent_method": "pcm"},
                "pcm": {"theory": "cpcm", "hpoints": "194"},
                "solvent": {"dielectric": "78.39"},
            },
        ),
        ("water_mol", {"rem": {"solvent_method": "smd"}, "smx": {"solvent": "water"}}),
        ("water_mol", {"scan": {"stre": ["1 2 0.95 1.35 0.1"]}}),
    ],
)
def test_overwrite(molecule, overwrite_inputs, request) -> None:
    """
    Test for ensuring whether overwrite_inputs correctly
    changes the default input_set parameters.

    Here, we use the StaticSetGenerator as an example,
    but any input generator that has a passed overwrite_inputs
    dict as an input argument could be used.
    """
    molecule = request.getfixturevalue(molecule)

    input_gen = SinglePointSetGenerator()
    input_gen.overwrite_inputs = overwrite_inputs
    in_set = input_gen.get_input_set(molecule)
    in_set_rem = {}
    in_set_opt = {}
    in_set_pcm = {}
    in_set_smx = {}
    in_set_solvent = {}
    in_set_scan = {}
    if overwrite_inputs.keys() == "rem":
        in_set_rem = in_set.qcinput.as_dict()["rem"]
    elif overwrite_inputs.keys() == "opt":
        in_set_opt = in_set.qcinput.as_dict()["opt"]
    elif overwrite_inputs.keys() == ["rem", "pcm", "solvent"]:
        in_set_rem = in_set.qcinput.as_dict()["rem"]
        in_set_pcm = in_set.qcinput.as_dict()["pcm"]
        in_set_solvent = in_set.qcinput.as_dict()["solvent"]
    elif overwrite_inputs.keys() == ["rem", "smx"]:
        in_set_rem = in_set.qcinput.as_dict()["rem"]
        in_set_smx = in_set.qcinput.as_dict()["smx"]

    if in_set_rem:  # case 1
        assert in_set_rem["method"] == "b97-d3"
        assert in_set_rem["dft_d"] == "d3_bj"
    elif in_set_opt:  # case 2
        assert in_set_opt["constraint"] == ["stre 1 2 0.96"]
    elif in_set_rem and in_set_pcm and in_set_solvent:
        assert in_set_rem["solvent_method"] == "pcm"
        assert in_set_pcm["theory"] == "cpcm"
        assert in_set_pcm["hpoints"] == "194"
        assert in_set_solvent["dielectric"] == "78.39"
    elif in_set_rem and in_set_smx:
        assert in_set_rem["solvent_method"] == "smd"
        assert in_set_smx["solvent"] == "water"
    elif in_set_scan:
        assert in_set_scan["stre"] == ["1 2 0.95 1.35 0.1"]


@pytest.mark.parametrize(
    "molecule",
    [("water_mol")],
)
def test_write_set(molecule, clean_dir, request) -> None:
    """
    Test for ensuring whether overwrite_inputs correctly
    changes the default input_set parameters.

    Here, we use the StaticSetGenerator as an example,
    but any input generator that has a passed overwrite_inputs
    dict as an input argument could be used.
    """
    molecule = request.getfixturevalue(molecule)

    input_gen = SinglePointSetGenerator()
    in_set = input_gen.get_input_set(molecule)
    in_set.write_input(directory="./inset_write", overwrite=True)
    chk_input_set = QCInputSet.from_directory(directory="./inset_write")
    assert os.path.isdir("./inset_write")
    assert isinstance(chk_input_set.qcinput, QCInput)
