import os
from datetime import datetime, timezone
from pathlib import Path

from jobflow import run_locally
from monty.io import zopen
from pymatgen.core import Structure
from pytest import approx, mark

from atomate2.common.jobs.utils import (
    remove_workflow_files,
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
    # Note that patches use `.post` suffix in MP DB versions
    db_version = stored_data["database_version"].split(".post")[0]
    datetime.strptime(db_version, "%Y.%m.%d").replace(tzinfo=timezone.utc)
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


def test_workflow_cleanup(tmp_dir):
    dirs = [Path(p) for p in ("test_1", "temp_2")]

    orig_files = ["foo.txt", "bar.txt.gz"]
    expected_file_list = []
    for _dir in dirs:
        assert not _dir.exists()
        _dir.mkdir(exist_ok=True, parents=True)
        assert _dir.is_dir()

        for f in orig_files:
            with zopen(_dir / f, "wt", encoding="utf8") as _f:
                _f.write(
                    "Lorem ipsum dolor sit amet,\n"
                    "consectetur adipiscing elit,\n"
                    "sed do eiusmod tempor incididunt\n"
                    "ut labore et dolore magna aliqua."
                )
            assert (_dir / f).is_file()
            assert os.path.getsize(_dir / f) > 0

            expected_file_list.append(_dir / f)

    job = remove_workflow_files(dirs, [f.split(".gz")[0] for f in orig_files])
    run_locally(job)
    assert all(not Path(f).is_file() for f in expected_file_list)
