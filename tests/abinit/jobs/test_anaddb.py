def test_anaddb_dfpt_dte_silicon_carbide_standard(
    mock_anaddb, abinit_test_dir, clean_dir
):
    import os

    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    from atomate2.abinit.schemas.anaddb import AnaddbTaskDoc

    # load the initial structure, the maker and the ref_paths from the test_dir
    test_dir = (
        abinit_test_dir
        / "jobs"
        / "anaddb"
        / "AnaddbDfptDteMaker"
        / "silicon_carbide_standard"
    )
    structure = Structure.from_file(test_dir / "initial_structure.json.gz")
    maker_info = loadfn(test_dir / "maker.json.gz")
    maker = maker_info["maker"]
    ref_paths = loadfn(test_dir / "ref_paths.json.gz")

    from pathlib import Path

    from monty.shutil import copy_r, decompress_dir, remove

    path_tmp_prev_outputs = Path(os.getcwd()) / "prev_outputs"
    if path_tmp_prev_outputs.exists():
        remove(path_tmp_prev_outputs)
    os.mkdir(path_tmp_prev_outputs)
    copy_r(src=test_dir / "prev_outputs", dst=path_tmp_prev_outputs)
    decompress_dir(path_tmp_prev_outputs)

    prev_outputs = [
        path_tmp_prev_outputs / subdir
        for subdir in next(os.walk(test_dir / "prev_outputs"))[1]
    ]

    mock_anaddb(ref_paths)

    # make the job, run it and ensure that it finished running successfully
    job = maker.make(structure=structure, prev_outputs=prev_outputs)
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, AnaddbTaskDoc)
