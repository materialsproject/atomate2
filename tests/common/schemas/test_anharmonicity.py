import json

import numpy as np
import pytest
from monty.json import MontyEncoder
from pydantic import ValidationError

from atomate2.common.schemas.anharmonicity import AnharmonicityDoc
from atomate2.common.schemas.phonons import PhononComputationalSettings


def test_anharmonicity_doc():
    kwargs = dict(
        total_dft_energy=None,
        supercell_matrix=np.eye(3),
        primitive_matrix=np.eye(3),
        code="test",
        phonopy_settings=PhononComputationalSettings(
            npoints_band=1, kpath_scheme="test", kpoint_density_dos=1
        ),
        thermal_displacement_data=None,
        jobdirs=None,
        uuids=None,
    )
    doc = AnharmonicityDoc(**kwargs)
    # Check validation raises no errors
    validated = AnharmonicityDoc.model_validate_json(json.dumps(doc, cls=MontyEncoder))
    assert isinstance(validated, AnharmonicityDoc)

    # Test invalid supercell matrix type fails
    with pytest.raises(ValidationError):
        doc = AnharmonicityDoc(**kwargs | {"supercell_matrix": (1, 1, 1)})

    # Test optional material_id
    doc = AnharmonicityDoc(**kwargs | {"material_id": 1234})
    assert doc.material_id == 1234

    # Test extra="allow" option
    doc = AnharmonicityDoc(**kwargs | {"extra_field": "test"})
    assert doc.extra_field == "test"
