def test_run_silicon_carbide_shg(mock_abinit, abinit_test_dir, clean_dir):
    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    # load the initial structure, the maker and the ref_paths from the test_dir
    test_dir = abinit_test_dir / "flows" / "dfpt" / "ShgFlowMaker" / "SiC_shg"
    structure = Structure.from_file(test_dir / "initial_structure.json.gz")
    maker_info = loadfn(test_dir / "maker.json.gz")
    maker = maker_info["maker"]
    ref_paths = loadfn(test_dir / "ref_paths.json.gz")

    mock_abinit(ref_paths)

    # make the flow or job, run it and ensure that it finished running successfully
    flow_or_job = maker.make(structure)
    responses = run_locally(flow_or_job, create_folders=True, ensure_success=True)

    # validation the outputs of the flow or job
    assert (
        len(responses) == 18
    )  # 1 scf + 3 ddk + 2 generate + 2 run_rf + 3 dde + 5 dte + 1 mrgddb + 1 anaddb
    for job, _ in flow_or_job.iterflow():
        if (
            job.name == "run_rf"
        ):  # maybe will change if Replace changes in definition of run_rf with output
            assert len(responses[job.uuid]) == 2
            continue
        assert len(responses[job.uuid]) == 1
