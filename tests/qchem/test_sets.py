import os

import pytest
from pymatgen.core.structure import Molecule
from pymatgen.io.qchem.inputs import QCInput
from pymatgen.io.qchem.sets import QCInputSet, SinglePointSetGenerator


@pytest.fixture(scope="module")
def water_mol() -> Molecule:
    """Dummy molecular structure for water as a test molecule."""
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.121], [-0.783, 0.0, -0.485], [0.783, 0.0, -0.485]],
    )


@pytest.mark.parametrize(
    "overwrite_inputs, expected",
    [
        (
            {"rem": {"ideriv": 1, "method": "B97-D3", "dft_d": "D3_BJ"}},
            {"rem": {"ideriv": "1", "method": "b97-d3", "dft_d": "d3_bj"}},
        ),
        (
            {"opt": {"CONSTRAINT": ["stre 1 2 0.96"]}},
            {"opt": {"constraint": ["stre 1 2 0.96"]}},
        ),
        (
            {
                "rem": {"solvent_method": "pcm"},
                "pcm": {"theory": "cpcm", "hpoints": "194"},
                "solvent": {"dielectric": "78.39"},
            },
            {
                "rem": {"solvent_method": "pcm"},
                "pcm": {"theory": "cpcm", "hpoints": "194"},
                "solvent": {"dielectric": "78.39"},
            },
        ),
        (
            {"rem": {"solvent_method": "smd"}, "smx": {"solvent": "water"}},
            {"rem": {"solvent_method": "smd"}, "smx": {"solvent": "water"}},
        ),
        (
            {"scan": {"stre": ["1 2 0.95 1.35 0.1"]}},
            {"scan": {"stre": ["1 2 0.95 1.35 0.1"]}},
        ),
    ],
)
def test_overwrite(overwrite_inputs: dict, expected: dict, water_mol: Molecule) -> None:
    """
    Test for ensuring whether overwrite_inputs correctly
    changes the default input_set parameters.

    Here, we use the SinglePointSetGenerator as an example,
    but any input generator that has a passed overwrite_inputs
    dict as an input argument could be used.
    """
    input_gen = SinglePointSetGenerator()
    default_rem = input_gen.get_input_set(water_mol).qcinput.rem

    input_gen.overwrite_inputs = overwrite_inputs
    qcinput = input_gen.get_input_set(water_mol).qcinput

    for section, settings in expected.items():
        section_dict = getattr(qcinput, section)
        for key, val in settings.items():
            assert section_dict[key] == val

    # everything that was not overwritten keeps its default value
    for key, val in default_rem.items():
        if key not in expected.get("rem", {}):
            assert qcinput.rem[key] == val


def test_write_set(water_mol: Molecule, clean_dir) -> None:
    """Test writing an input set to a directory and reading it back."""
    input_gen = SinglePointSetGenerator()
    in_set = input_gen.get_input_set(water_mol)
    in_set.write_input(directory="./inset_write", overwrite=True)
    assert os.path.isdir("./inset_write")
    assert os.listdir("./inset_write") == ["mol.qin"]

    chk_input_set = QCInputSet.from_directory(directory="./inset_write")
    assert isinstance(chk_input_set, QCInputSet)
    assert isinstance(chk_input_set.qcinput, QCInput)
    assert str(chk_input_set.qcinput) == str(in_set.qcinput)

    with pytest.raises(FileExistsError):
        in_set.write_input(directory="./inset_write", overwrite=False)
