import json

import numpy as np
import pytest
from monty.json import MontyEncoder
from phonopy import Phonopy
from pydantic import ValidationError
from pymatgen.core import Lattice, Structure
from pymatgen.io.phonopy import get_phonopy_structure

from atomate2.common.schemas.phonons import (
    PhononBSDOSDoc,
    PhononComputationalSettings,
    PhononJobDirs,
    PhononUUIDs,
    ThermalDisplacementData,
    _set_nac_params,
)


def test_thermal_displacement_data():
    doc = ThermalDisplacementData(freq_min_thermal_displacements=0.0)
    validated = ThermalDisplacementData.model_validate_json(
        json.dumps(doc, cls=MontyEncoder)
    )
    assert isinstance(validated, ThermalDisplacementData)


def test_phonon_bs_dos_doc():
    kwargs = {
        "total_dft_energy": None,
        "supercell_matrix": np.eye(3),
        "primitive_matrix": np.eye(3),
        "code": "test",
        "phonopy_settings": PhononComputationalSettings(
            npoints_band=1, kpath_scheme="test", kpoint_density_dos=1
        ),
        "thermal_displacement_data": None,
        "jobdirs": None,
        "uuids": None,
    }
    doc = PhononBSDOSDoc(**kwargs)
    # check validation raises no errors
    validated = PhononBSDOSDoc.model_validate_json(json.dumps(doc, cls=MontyEncoder))
    assert isinstance(validated, PhononBSDOSDoc)

    # test invalid supercell_matrix type fails
    with pytest.raises(ValidationError):
        doc = PhononBSDOSDoc(**kwargs | {"supercell_matrix": (1, 1, 1)})

    # test optional material_id
    doc = PhononBSDOSDoc(**kwargs | {"material_id": 1234})
    assert doc.material_id == 1234

    # test extra="allow" option
    doc = PhononBSDOSDoc(**kwargs | {"extra_field": "test"})
    assert doc.extra_field == "test"


# schemas where all fields have default values
@pytest.mark.parametrize("model_cls", [PhononJobDirs, PhononUUIDs])
def test_model_validate(model_cls):
    validated = model_cls.model_validate_json(json.dumps(model_cls(), cls=MontyEncoder))
    assert isinstance(validated, model_cls)


def get_nacl_phonon():
    structure = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.6), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).get_primitive_structure()
    return Phonopy(get_phonopy_structure(structure), supercell_matrix=np.eye(3))


@pytest.mark.parametrize(
    ("code", "charge", "has_nac"),
    [
        ("vasp", 1, True),
        ("forcefields", 1, True),
        ("ase", 1, True),
        ("torchsim", 1, True),
        ("aims", 1, False),
        ("vasp", 0, False),
    ],
)
def test_set_nac_params(code, charge, has_nac):
    phonon = get_nacl_phonon()
    born = charge * np.array([1, -1])[:, None, None] * np.eye(3)
    borns, epsilon = _set_nac_params(
        phonon, born.tolist(), (2 * np.eye(3)).tolist(), 1e-4, code
    )

    assert borns == pytest.approx(born)
    assert epsilon == pytest.approx(2 * np.eye(3))
    assert (phonon.nac_params is not None) == has_nac


def test_set_nac_params_without_born():
    phonon = get_nacl_phonon()
    assert _set_nac_params(phonon, None, None, 1e-4, "vasp") == (None, None)
    assert phonon.nac_params is None
    with pytest.raises(ValueError, match="Number of Born charges"):
        _set_nac_params(phonon, [np.eye(3).tolist()], np.eye(3).tolist(), 1e-4, "vasp")
