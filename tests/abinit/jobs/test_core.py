from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jobflow import Maker


def test_static_run_silicon_standard(mock_abinit, abinit_test_dir, clean_dir):
    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    from atomate2.abinit.schemas.task import AbinitTaskDoc as AbinitTaskDocument

    # load the initial structure, the maker and the ref_paths from the test_dir
    test_dir = abinit_test_dir / "jobs" / "core" / "StaticMaker" / "silicon_standard"
    structure = Structure.from_file(test_dir / "initial_structure.json.gz")
    maker: Maker = loadfn(test_dir / "maker.json.gz")["maker"]
    ref_paths = loadfn(test_dir / "ref_paths.json.gz")

    mock_abinit(ref_paths)

    # make the job, run it and ensure that it finished running successfully
    job = maker.make(structure)
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, AbinitTaskDocument)
    # assert output1.run_number == 1


def test_static_run_silicon_restarts(mock_abinit, abinit_test_dir, clean_dir):
    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    from atomate2.abinit.schemas.task import AbinitTaskDoc as AbinitTaskDocument

    # load the initial structure, the maker and the ref_paths from the test_dir
    test_dir = abinit_test_dir / "jobs" / "core" / "StaticMaker" / "silicon_restarts"
    structure = Structure.from_file(test_dir / "initial_structure.json.gz")
    maker: Maker = loadfn(test_dir / "maker.json.gz")["maker"]
    ref_paths = loadfn(test_dir / "ref_paths.json.gz")

    mock_abinit(ref_paths)

    # make the job, run it and ensure that it finished running successfully
    job = maker.make(structure)
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, AbinitTaskDocument)
    # assert output1.run_number == 1
    # TODO 2024-04-25: figure out why responses[job.uuid][2] causes KeyError
    # output2 = responses[job.uuid][2].output
    # assert isinstance(output2, AbinitTaskDocument)
    # assert output2.run_number == 2


def test_relax_run_silicon_scaled1p2_standard(mock_abinit, abinit_test_dir, clean_dir):
    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    from atomate2.abinit.schemas.task import AbinitTaskDoc as AbinitTaskDocument

    # load the initial structure, the maker and the ref_paths from the test_dir
    test_dir = (
        abinit_test_dir / "jobs" / "core" / "RelaxMaker" / "silicon_scaled1p2_standard"
    )
    structure = Structure.from_file(test_dir / "initial_structure.json.gz")
    maker: Maker = loadfn(test_dir / "maker.json.gz")["maker"]
    ref_paths = loadfn(test_dir / "ref_paths.json.gz")

    mock_abinit(ref_paths)

    # make the flow or job, run it and ensure that it finished running successfully
    job = maker.make(structure)
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the flow or job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, AbinitTaskDocument)
    # assert output1.run_number == 1


def test_relax_run_silicon_scaled1p2_restart(mock_abinit, abinit_test_dir, clean_dir):
    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    from atomate2.abinit.schemas.task import AbinitTaskDoc as AbinitTaskDocument

    # load the initial structure, the maker and the ref_paths from the test_dir
    test_dir = (
        abinit_test_dir / "jobs" / "core" / "RelaxMaker" / "silicon_scaled1p2_restart"
    )
    structure = Structure.from_file(test_dir / "initial_structure.json.gz")
    maker: Maker = loadfn(test_dir / "maker.json.gz")["maker"]
    ref_paths = loadfn(test_dir / "ref_paths.json.gz")

    mock_abinit(ref_paths)

    # make the flow or job, run it and ensure that it finished running successfully
    job = maker.make(structure)
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the flow or job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, AbinitTaskDocument)
    # assert output1.run_number == 1
    # TODO 2024-04-25: figure out why responses[job.uuid][2] causes KeyError
    # output2 = responses[job.uuid][2].output
    # assert isinstance(output2, AbinitTaskDocument)
    # assert output2.run_number == 2
