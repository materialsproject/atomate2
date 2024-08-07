import json

import numpy as np
import pytest
from monty.json import MontyEncoder
from pydantic import ValidationError

from atomate2.common.schemas.phonons import (
    PhononBSDOSDoc,
    PhononComputationalSettings,
    PhononJobDirs,
    PhononUUIDs,
    ThermalDisplacementData,
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
