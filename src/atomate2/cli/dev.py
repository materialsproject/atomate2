"""Module containing command line scripts for developers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click
from monty.json import jsanitize

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Maker


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

    task_labels = [
        o.output.task_label for o in outputs if isinstance(o.output, TaskDoc)
    ]

    if len(task_labels) != len(set(task_labels)):
        raise ValueError("Not all jobs have unique names")

    original_mapping = {}
    mapping = {}
    for output in outputs:
        if not isinstance(output.output, TaskDoc):
            # this is not a VASP job
            continue

        job_name = output.output.task_label
        orig_job_dir = strip_hostname(output.output.dir_name)
        folder_name = output.output.task_label.replace("/", "_").replace(" ", "_")

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

    run_vasp_kwargs = {key: {"incar_settings": ["NSW", "ISMEAR"]} for key in mapping}
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


@dev.command(context_settings={"help_option_names": ["-h", "--help"]})
def abinit_script_maker() -> None:
    """Generate template script for abinit makers."""
    import os
    import sys

    script_fname = "create_maker.py"
    if os.path.exists(script_fname):
        click.echo(f"{script_fname} already exists, refusing to overwrite it")
        sys.exit()
    out = [
        "from atomate2.abinit.flows.core import BandStructureMaker",
        "from atomate2.abinit.jobs.core import StaticMaker, LineNonSCFMaker",
        "from atomate2.abinit.powerups import update_factory_kwargs, update_generator_attributes",  # noqa: E501
        "from atomate2.cli.dev import save_abinit_maker",
        "",
        "",
        "# The following lines define the maker.",
        "# Adapt for specific job/flow maker test.",
        "maker = BandStructureMaker()",
        "maker = update_factory_kwargs(maker, dict(kppa=10, ecut=4.0, nband=4), class_filter=StaticMaker)",  # noqa: E501
        "maker = update_factory_kwargs(maker, dict(ndivsm=2), class_filter=LineNonSCFMaker)",  # noqa: E501
        "maker = update_generator_attributes(maker, dict(nbands_factor=1.5), class_filter=LineNonSCFMaker)",  # noqa: E501
        "",
        "# Save the maker and metadata to maker.json",
        "save_abinit_maker(maker)",
        "",
    ]
    with open(script_fname, "w") as file:
        file.write("\n".join(out))


@dev.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("structure_file", required=False)
@click.option("--make-kwargs", "-mk", required=False)
def abinit_generate_reference(structure_file: str, make_kwargs: dict) -> None:
    """Generate reference files for abinit test.

    This script expects there is a maker.json file and a structure file. The structure
    file can either be "initial_structure.json" or another file that can be opened by
    pymatgen. Keyword arguments can also be passed down to the Maker's make method
    through the make_kwargs option.

    NB: note that all the files in the folder will be compressed, in light of a copy to
    the test folder, unless the --no-compress is selected.
    """
    from jobflow import JobStore, run_locally
    from maggma.stores.mongolike import MemoryStore
    from monty.serialization import dumpfn, loadfn
    from pymatgen.core.structure import Structure

    # get the structure
    if structure_file:
        struct = Structure.from_file(filename=structure_file)
        struct.to(filename="initial_structure.json")
    else:
        struct = Structure.from_file(filename="initial_structure.json")

    # initialize the maker from the maker.json file
    maker_info = loadfn("maker.json")
    maker = maker_info["maker"]

    # make the flow and dump the keyword arguments passed to the Maker's make method
    # if it is not empty
    make_kwargs = make_kwargs or {}
    flow_or_job = maker.make(structure=struct, **make_kwargs)
    if make_kwargs:
        dumpfn({"make_kwargs": make_kwargs}, "make_info.json")

    # run the workflow using a custom store so that we can easily compile test data
    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    run_locally(flow_or_job, store=store, create_folders=True)

    # dump all of the job outputs to the outputs.json file in the current directory
    outputs = list(store.query(load=True))
    dumpfn(outputs, "outputs.json")


@dev.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("test_name")
@click.option("--test-data-dir", required=False)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Whether to force overwrite of existing folders/files.",
)
def abinit_test_data(test_name: str, test_data_dir: str | None, force: bool) -> None:
    """Copy test data for ABINIT unit tests.

    This script expects there is a maker.json file, an initial_structure.json file,
    an outputs.json file and job folders in the current directory. Please refer to
    the atomate2 documentation on writing unit tests for more information.
    """
    import sys
    import warnings
    from pathlib import Path

    from monty.serialization import dumpfn, loadfn
    from monty.shutil import compress_dir

    from atomate2.abinit.schemas.core import AbinitTaskDocument
    from atomate2.common.files import copy_files
    from atomate2.utils.path import strip_hostname

    if test_data_dir is None:
        abinit_test_data_dir = (
            Path(__file__).parent
            / ".."
            / ".."
            / ".."
            / "tests"
            / "test_data"
            / "abinit"
        )
    else:
        abinit_test_data_dir = Path(test_data_dir) / "abinit"

    if not abinit_test_data_dir.exists():
        click.echo(
            "The following test_data/abinit "
            f"directory does not exist: {abinit_test_data_dir}"
        )
        sys.exit()

    warnings.filterwarnings("ignore", module="pymatgen")

    maker_info = loadfn("maker.json")
    maker = maker_info["maker"]

    maker_name = maker.__class__.__name__
    # take the module path and exclude the first two elements
    # (i.e. "atomate2" and "abinit")
    module_path = maker.__module__.split(".")[2:]
    maker_dir = abinit_test_data_dir / Path(*module_path) / maker_name
    if not maker_dir.exists():
        maker_dir.mkdir(parents=True)
    test_dir = maker_dir / test_name

    def _makedir(directory: Path, force_overwrite: bool) -> None:
        if not directory.exists():
            directory.mkdir()
        elif not force_overwrite:
            click.echo(f"folder ({directory}) already exists, refusing to overwrite it")
            sys.exit()

    _makedir(test_dir, force_overwrite=force)

    outputs = loadfn("outputs.json")

    task_labels = [
        o["output"].task_label for o in outputs if isinstance(o, AbinitTaskDocument)
    ]

    if len(task_labels) != len(set(task_labels)):
        raise ValueError("Not all jobs have unique names")

    original_mapping = {}
    mapping: dict[Any, Any] = {}

    def _fake_dirdata(
        src_dir: Path,
        dest_dir: Path,
        dirdata_name: str,
        include_files: list | None = None,
        include_fake_files: list | None = None,
        force_overwrite: bool = False,
        allow_missing: bool = True,
    ) -> None:
        src_dirdata = src_dir / dirdata_name
        if not src_dirdata.exists():
            raise RuntimeError(f"Source directory {src_dirdata} does not exist")
        dest_dirdata = dest_dir / dirdata_name
        _makedir(dest_dirdata, force_overwrite=force_overwrite)
        empty_file = dest_dirdata / ".empty"
        empty_file.write_text(
            "Empty file for git to be able to have an empty directory"
        )
        if include_files:
            copy_files(
                src_dirdata,
                dest_dirdata,
                include_files=include_files,
                allow_missing=allow_missing,
            )
        if include_fake_files:
            for fname in include_fake_files:
                src_fpath = src_dirdata / fname
                dest_fpath = dest_dirdata / fname
                if not src_fpath.exists():
                    if allow_missing:
                        continue
                    raise RuntimeError(f"File {src_fpath} does not exist")
                if src_fpath.is_symlink():
                    dest_fpath.write_text(f"SYMBOLIC_LINK TO {src_fpath.resolve()}")
                elif src_fpath.is_file():
                    dest_fpath.write_text("REGULAR_FILE")
                else:
                    raise RuntimeError(
                        "File is not a symbolic link nor a regular file."
                    )

    def _fake_dirs(
        src_dir: Path,
        dest_dir: Path,
        indata_files: list | None = None,
        outdata_files: list | None = None,
        tmpdata_files: list | None = None,
        indata_fake_files: list | None = None,
        outdata_fake_files: list | None = None,
        tmpdata_fake_files: list | None = None,
        force_overwrite: bool = False,
        allow_missing: bool = True,
    ) -> None:
        for dirdata_name, data_files, data_fake_files in zip(
            ("indata", "outdata", "tmpdata"),
            (indata_files, outdata_files, tmpdata_files),
            (indata_fake_files, outdata_fake_files, tmpdata_fake_files),
        ):
            _fake_dirdata(
                src_dir=src_dir,
                dest_dir=dest_dir,
                dirdata_name=dirdata_name,
                include_files=data_files,
                include_fake_files=data_fake_files,
                force_overwrite=force_overwrite,
                allow_missing=allow_missing,
            )

    for output in outputs:
        if not isinstance(output["output"], AbinitTaskDocument):
            # this is not an Abinit job
            continue

        job_name = output["output"].task_label
        orig_job_dir = Path(strip_hostname(output["output"].dir_name))
        folder_name = output["output"].task_label.replace("/", "_").replace(" ", "_")
        index_job = str(output["index"])

        job_dir = test_dir / folder_name
        _makedir(job_dir, force_overwrite=force)
        job_index_dir = job_dir / index_job
        _makedir(job_index_dir, force_overwrite=force)

        job_index_dir = job_index_dir.resolve()

        job_index_dir_parts = job_index_dir.parts
        indices = _find_sub_list(
            (
                "atomate2",
                "tests",
                "test_data",
                "abinit",
            ),
            job_index_dir_parts,
        )
        if len(indices) != 1:
            raise RuntimeError(
                "Could not find job_test's "
                "atomate2/tests/test_data/abinit-based root dir"
            )
        job_index_dir_for_mapping = Path(*job_index_dir_parts[indices[0][1] :])
        if job_name not in mapping:
            mapping[job_name] = {}
        if index_job in mapping[job_name]:
            raise RuntimeError(
                f"Job index {index_job} for "
                f"{job_name} Job is already present in the mapping"
            )
        mapping[job_name][index_job] = str(job_index_dir_for_mapping)
        original_mapping[str(job_index_dir)] = orig_job_dir

        # create input folder and copy across files
        input_dir = job_index_dir / "inputs"
        _makedir(input_dir, force_overwrite=force)

        copy_files(
            orig_job_dir,
            input_dir,
            include_files=[
                "run.abi",
                "abinit_input.json",
            ],
            allow_missing=False,
        )
        _fake_dirs(
            src_dir=orig_job_dir,
            dest_dir=input_dir,
            indata_files=None,
            outdata_files=None,
            tmpdata_files=None,
            indata_fake_files=["in_DEN", "in_WFK"],
            outdata_fake_files=None,
            tmpdata_fake_files=None,
            force_overwrite=force,
            allow_missing=True,
        )
        # create output folder and copy across files
        output_dir = job_index_dir / "outputs"
        _makedir(output_dir, force_overwrite=force)
        copy_files(
            orig_job_dir,
            output_dir,
            include_files=[
                "run.abo",
                "run.err",
                "run.log",
            ],
            allow_missing=False,
        )
        _fake_dirs(
            src_dir=orig_job_dir,
            dest_dir=output_dir,
            indata_files=None,
            outdata_files=["out_GSR.nc", "out_FATBANDS.nc"],
            tmpdata_files=None,
            indata_fake_files=None,
            outdata_fake_files=["out_DEN", "out_WFK"],
            tmpdata_fake_files=None,
            force_overwrite=force,
            allow_missing=True,
        )

    dumpfn(mapping, "ref_paths.json")
    flow_files = ["maker.json", "initial_structure.json", "ref_paths.json"]
    if Path("make_info.json").exists():
        flow_files.append("make_info.json")
    copy_files(".", test_dir, include_files=flow_files)
    original_mapping_str = "\n".join(
        [f"  {v}  ->  {k}" for k, v in original_mapping.items()]
    )

    compress_dir(test_dir)

    # starting index of the parts of the test_dir that needs to be used in the
    # example script. Looks for the last occurrence of the "abinit" part of the path.
    for i, p in enumerate(reversed(test_dir.parts)):
        if p == "abinit":
            index_part = -i
            break
    else:
        raise RuntimeError("Did not find 'abinit' in the test path")
    test_function_str = f"""Test files generated in test_data.

Please ensure that all other necessary files are included in the test_data, such as
run.abi, run.abo, run.log, etc. as well as indata, outdata and tmpdata directories
with their respective relevant reference files.
Include additional files only if they are absolutely necessary otherwise they will
increase the size of the atomate2 repository.

A mapping from the original job folders to the formatted folders is:
{original_mapping_str}

An example test using the generated test data is provided below. Be sure to update
this test with the particulars for your system. For more examples, see the existing
tests in atomate2/tests/abinit/jobs and atomate2/tests/abinit/flows.

class Test{maker_name}:
    def test_run_{test_name}(self, mock_abinit, abinit_test_dir, clean_dir):
        from jobflow import run_locally
        from pymatgen.core.structure import Structure
        from monty.serialization import loadfn
        from atomate2.abinit.schemas.core import AbinitTaskDocument

        # load the initial structure, the maker and the ref_paths from the test_dir
        test_dir = abinit_test_dir / {" / ".join(
        [f'"{part}"' for part in test_dir.parts[index_part:]]
    )}
        structure = Structure.from_file(test_dir / "initial_structure.json.gz")
        maker_info = loadfn(test_dir / "maker.json.gz")
        maker = maker_info["maker"]
        ref_paths = loadfn(test_dir / "ref_paths.json.gz")

        mock_abinit(ref_paths)

        # make the flow or job, run it and ensure that it finished running successfully
        flow_or_job = maker.make(structure)
        responses = run_locally(flow_or_job, create_folders=True, ensure_success=True)

        # validation the outputs of the flow or job
        output1 = responses[flow_or_job.uuid][1].output
        assert isinstance(output1, AbinitTaskDocument)
        assert output1.structure == structure
        assert output1.run_number == 1
    """

    print(test_function_str)  # noqa: T201


def save_abinit_maker(maker: Maker) -> None:
    """Save maker, the script used to create it and additional metadata."""
    import inspect
    import json
    import shutil
    import subprocess
    import warnings
    from datetime import datetime, timezone

    caller_frame = inspect.stack()[1]
    caller_filename_full = caller_frame.filename
    with open(caller_filename_full) as file:
        script_str = file.read()
    git = shutil.which("git")
    author = None
    author_mail = None
    if git:
        name = subprocess.run(
            "git config user.name".split(),
            capture_output=True,
            encoding="utf-8",
            check=True,
        )
        mail = subprocess.run(
            "git config user.email".split(),
            capture_output=True,
            encoding="utf-8",
            check=True,
        )
        if name.returncode == 0:
            author = name.stdout.strip()
        else:
            warnings.warn(
                "Author could not be detected from git. "
                "You may want to manually set it in the 'maker.json' file.",
                stacklevel=2,
            )
        if mail.returncode == 0:
            author_mail = mail.stdout.strip()
        else:
            warnings.warn(
                "Author email could not be detected from git. "
                "You may want to manually set it in the 'maker.json' file.",
                stacklevel=2,
            )
    with open("maker.json", "w") as file:
        json.dump(
            {
                "author": author,
                "author_mail": author_mail,
                "created_on": str(datetime.now(tz=timezone.utc)),
                "maker": jsanitize(maker.as_dict()),
                "script": script_str,
            },
            file,
        )


def _find_sub_list(sublist: list | tuple, mainlist: list | tuple) -> list:
    sll = len(sublist)
    inds = [i for i, e in enumerate(mainlist) if e == sublist[0]]

    return [(ind, ind + sll) for ind in inds if mainlist[ind : ind + sll] == sublist]
