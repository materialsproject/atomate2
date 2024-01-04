import json

import numpy as np
import pytest
from monty.json import MontyEncoder

from atomate2.common.schemas.phonons import (
    PhononBSDOSDoc,
    PhononComputationalSettings,
    PhononJobDirs,
    PhononUUIDs,
    ThermalDisplacementData,
)


def test_thermal_displacement_data():
    doc = ThermalDisplacementData(freq_min_thermal_displacements=0.0)
    ThermalDisplacementData.model_validate_json(json.dumps(doc, cls=MontyEncoder))


def test_phonon_bsdos_doc():
    dummy_matrix_3d = tuple([np.array(x) for x in np.eye(3).tolist()])
    doc = PhononBSDOSDoc(
        total_dft_energy=None,
        supercell_matrix=dummy_matrix_3d,
        primitive_matrix=dummy_matrix_3d,
        code="test",
        phonopy_settings=PhononComputationalSettings(
            npoints_band=1, kpath_scheme="test", kpoint_density_dos=1
        ),
        thermal_displacement_data=None,
        jobdirs=None,
        uuids=None,
    )
    PhononBSDOSDoc.model_validate_json(json.dumps(doc, cls=MontyEncoder))


# schemas where all fields have default values
@pytest.mark.parametrize(
    "model_cls",
    [
        PhononJobDirs,
        PhononUUIDs,
    ],
)
def test_model_validate(model_cls):
    model_cls.model_validate_json(json.dumps(model_cls(), cls=MontyEncoder))
