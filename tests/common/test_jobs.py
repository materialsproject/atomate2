import os
from datetime import datetime, timezone

from jobflow import run_locally
from pymatgen.core import Structure
from pytest import approx, mark

from atomate2.common.jobs.utils import (
    retrieve_structure_from_materials_project,
    structure_to_conventional,
    structure_to_primitive,
)


def test_structure_to_primitive(si_structure):
    job = structure_to_primitive(si_structure)

    responses = run_locally(job)
    output = responses[job.uuid][1].output

    assert output.lattice.alpha == approx(60)


def test_structure_to_conventional(si_structure):
    job = structure_to_conventional(si_structure)

    responses = run_locally(job)
    output = responses[job.uuid][1].output

    assert output.lattice.alpha == approx(90)


@mark.skipif(
    not os.getenv("MP_API_KEY"),
    reason="Materials Project API key not set in environment.",
)
def test_retrieve_structure_from_materials_project():
    job = retrieve_structure_from_materials_project("mp-149")

    responses = run_locally(job)
    output = responses[job.uuid][1].output
    stored_data = responses[job.uuid][1].stored_data

    assert isinstance(output, Structure)

    # test stored data is in expected format
    datetime.strptime(stored_data["database_version"], "%Y.%m.%d").replace(
        tzinfo=timezone.utc
    )
    assert stored_data["task_id"].startswith("mp-")

    job = retrieve_structure_from_materials_project(
        "mp-19009", reset_magnetic_moments=False
    )

    responses = run_locally(job)
    output = responses[job.uuid][1].output

    assert "magmom" in output.site_properties

    job = retrieve_structure_from_materials_project(
        "mp-19009", reset_magnetic_moments=True
    )

    responses = run_locally(job)
    output = responses[job.uuid][1].output

    assert "magmom" not in output.site_properties
