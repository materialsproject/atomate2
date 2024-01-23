import pytest
from pymatgen.core.structure import Molecule

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
    if overwrite_inputs.keys() == "rem":
        in_set_rem = in_set.qcinput.as_dict()["rem"]
    elif overwrite_inputs.keys() == "opt":
        in_set_opt = in_set.qcinput.as_dict()["opt"]

    if in_set_rem:  # case 1
        assert in_set_rem["method"] == "b97-d3"
        assert in_set_rem["dft_d"] == "d3_bj"
    elif in_set_opt:  # case 2
        assert in_set_opt["constraint"] == ["stre 1 2 0.96"]
