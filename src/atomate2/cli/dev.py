"""Module containing command line scripts for developers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from pathlib import Path


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def dev() -> None:
    """Tools for atomate2 developers."""


@dev.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "test_dir",
)
@click.option(
    "--additional_file",
    "-a",
    multiple=True,
    help="list of additional files to copy from each completed VASP directory. "
    "Example: `--additional_file CHGCAR --additional_file LOCPOT`",
)
def vasp_test_data(test_dir: str | Path, additional_file: list[str]) -> None:
    """Generate test data for VASP unit tests.

    This script expects there is an outputs.json file and job folders in the current
    directory. Please refer to the atomate2 documentation on writing unit tests for more
    information.

    Parameters
    ----------
    test_dir
        The directory to write the test data to.
        Should not contain spaces or punctuation.
    additional_files
        list of additional files to copy from each completed VASP directory.
        Example: `--additional_file CHGCAR --additional_file LOCPOT`,
    """
    import warnings
    from pathlib import Path
    from pprint import pformat

    from emmet.core.tasks import TaskDoc
    from monty.serialization import loadfn

    from atomate2.common.files import copy_files, delete_files, gunzip_files
    from atomate2.utils.path import strip_hostname

    warnings.filterwarnings("ignore", module="pymatgen")

    test_dir = Path(test_dir)

    if test_dir.exists():
        click.echo("test_data folder already exists, refusing to overwrite it")
        raise SystemExit(1)

    test_dir.mkdir()

    outputs = loadfn("outputs.json")

    task_labels = [o["output"].task_label for o in outputs if isinstance(o, TaskDoc)]

    if len(task_labels) != len(set(task_labels)):
        raise ValueError("Not all jobs have unique names")

    original_mapping = {}
    mapping = {}
    for output in outputs:
        if not isinstance(output["output"], TaskDoc):
            # this is not a VASP job
            continue

        job_name = output["output"].task_label
        orig_job_dir = strip_hostname(output["output"].dir_name)
        folder_name = output["output"].task_label.replace("/", "_").replace(" ", "_")

        if len(task_labels) == 1:
            # only testing a single job
            job_dir = test_dir
        else:
            job_dir = test_dir / folder_name
            job_dir.mkdir()

        mapping[job_name] = str(job_dir)
        original_mapping[str(job_dir)] = orig_job_dir

        # create input folder and copy across files
        input_dir = job_dir / "inputs"
        input_dir.mkdir()

        copy_files(
            orig_job_dir,
            input_dir,
            include_files=[
                "INCAR",
                "INCAR.gz",
                "KPOINTS",
                "KPOINTS.gz",
                "POTCAR",
                "POTCAR.gz",
                "POSCAR",
                "POSCAR.gz",
            ],
            allow_missing=True,
        )
        gunzip_files(input_dir)
        _potcar_to_potcar_spec(input_dir / "POTCAR", input_dir / "POTCAR.spec")
        delete_files(input_dir, include_files=["POTCAR"])

        # create output folder and copy across files
        output_dir = job_dir / "outputs"
        output_dir.mkdir()
        copy_files(
            orig_job_dir,
            output_dir,
            include_files=[
                "POSCAR*",
                "CONTCAR*",
                "KPOINTS*",
                "INCAR*",
                "vasprun*",
                "OUTCAR*",
                "*.json*",
                *additional_file,
            ],
            allow_missing=True,
        )
        copy_files(input_dir, output_dir, include_files=["POTCAR.spec"])

    mapping_str = pformat(mapping).replace("\n", "\n    ")
    original_mapping_str = "\n".join(
        [f"  {v}  ->  {k}" for k, v in original_mapping.items()]
    )

    run_vasp_kwargs = {k: {"incar_settings": ["NSW", "ISMEAR"]} for k in mapping}
    run_vasp_kwargs_str = pformat(run_vasp_kwargs).replace("\n", "\n    ")

    test_function_str = f"""Test files generated in test_data.

Please ensure that all other necessary files are included in the test_data, such as
CHGCAR, LOCPOT, etc. However, only include additional files if they are absolutely
necessary (for example to calculate defect corrections) otherwise they will increase
the size of the atomate2 repository.

A mapping from the original job folders to the formatted folders is:
{original_mapping_str}

An example test using the generated test data is provided below. Be sure to update
this test with the particulars for your system. For more examples, see the existing
tests in atomate2/tests/vasp/jobs.

def test_my_flow(mock_vasp, clean_dir, si_structure):
    from jobflow import run_locally
    from emmet.core.tasks import TaskDoc

    # mapping from job name to directory containing test files
    ref_paths = {mapping_str}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {run_vasp_kwargs_str}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    job = MyMaker().make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate the outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert output1.output.energy == pytest.approx(-10.85037078)
    """

    print(test_function_str)  # noqa: T201


def _potcar_to_potcar_spec(potcar_filename: str | Path, output_filename: Path) -> None:
    """Convert a POTCAR file to a POTCAR.spec file."""
    from pymatgen.io.vasp import Potcar

    potcar = Potcar.from_file(potcar_filename)
    output_filename.write_text("\n".join(potcar.symbols))
