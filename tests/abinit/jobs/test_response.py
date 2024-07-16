def test_ddk_run_silicon_carbide_standard(mock_abinit, abinit_test_dir, clean_dir):
    import os

    from jobflow import run_locally
    from monty.serialization import loadfn

    from atomate2.abinit.schemas.task import AbinitTaskDoc

    # load the initial structure, the maker and the ref_paths from the test_dir
    test_dir = (
        abinit_test_dir / "jobs" / "response" / "DdkMaker" / "silicon_carbide_standard"
    )
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
        #    test_dir / "prev_outputs" / subdir
        path_tmp_prev_outputs / subdir
        for subdir in next(os.walk(test_dir / "prev_outputs"))[1]
    ]

    mock_abinit(ref_paths)

    # make the job, run it and ensure that it finished running successfully
    job = maker.make(prev_outputs=prev_outputs)
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, AbinitTaskDoc)


def test_dde_run_silicon_carbide_standard(mock_abinit, abinit_test_dir, clean_dir):
    import os

    from jobflow import run_locally
    from monty.serialization import loadfn

    from atomate2.abinit.schemas.task import AbinitTaskDoc

    # load the initial structure, the maker and the ref_paths from the test_dir
    test_dir = (
        abinit_test_dir / "jobs" / "response" / "DdeMaker" / "silicon_carbide_standard"
    )
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
        #    test_dir / "prev_outputs" / subdir
        path_tmp_prev_outputs / subdir
        for subdir in next(os.walk(test_dir / "prev_outputs"))[1]
    ]

    mock_abinit(ref_paths)

    # make the job, run it and ensure that it finished running successfully
    job = maker.make(prev_outputs=prev_outputs)
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, AbinitTaskDoc)
