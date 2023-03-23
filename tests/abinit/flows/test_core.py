def test_band_structure_run_silicon(mock_abinit, abinit_test_dir, clean_dir):
    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    # load the initial structure, the maker and the ref_paths from the test_dir
    test_dir = abinit_test_dir / "jobs" / "BandStructureMaker" / "silicon"
    structure = Structure.from_file(test_dir / "initial_structure.json")
    maker_info = loadfn(test_dir / "maker.json")
    maker = maker_info["maker"]
    ref_paths = loadfn(test_dir / "ref_paths.json")

    mock_abinit(ref_paths)

    # make the flow or job, run it and ensure that it finished running successfully
    flow_or_job = maker.make(structure)
    responses = run_locally(flow_or_job, create_folders=True, ensure_success=True)

    # validation the outputs of the flow or job
    assert len(responses) == 3
    for job, _parents in flow_or_job.iterflow():
        assert len(responses[job.uuid]) == 1
