import json
import os

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


def test_check_phonon_output_filenames(clean_dir, si_structure):
    from jobflow import run_locally

    from atomate2.forcefields.flows.phonons import PhononMaker
    from atomate2.forcefields.jobs import CHGNetRelaxMaker, CHGNetStaticMaker

    phojob = PhononMaker(
        bulk_relax_maker=CHGNetRelaxMaker(steps=5),
        phonon_displacement_maker=CHGNetStaticMaker(),
        static_energy_maker=CHGNetStaticMaker(),
        store_force_constants=False,
        generate_frequencies_eigenvectors_kwargs={
            "filename_BS": "phonon_BS_test.png",
            "filename_DOS": "phonon_DOS_test.pdf",
        },
    ).make(structure=si_structure)

    run_locally(phojob, ensure_success=True)

    files = os.listdir(os.getcwd())

    for png in [file for file in files if file.endswith(".png")]:
        assert png == "phonon_BS_test.png"

    for pdf in [file for file in files if file.endswith(".pdf")]:
        assert pdf == "phonon_DOS_test.pdf"
