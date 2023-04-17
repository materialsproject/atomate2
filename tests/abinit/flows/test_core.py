def test_band_structure_run_silicon(mock_abinit, abinit_test_dir, clean_dir):
    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    # load the initial structure, the maker and the ref_paths from the test_dir
    test_dir = abinit_test_dir / "flows" / "core" / "BandStructureMaker" / "silicon"
    structure = Structure.from_file(test_dir / "initial_structure.json.gz")
    maker_info = loadfn(test_dir / "maker.json.gz")
    maker = maker_info["maker"]
    ref_paths = loadfn(test_dir / "ref_paths.json.gz")

    mock_abinit(ref_paths)

    # make the flow or job, run it and ensure that it finished running successfully
    flow_or_job = maker.make(structure)
    responses = run_locally(flow_or_job, create_folders=True, ensure_success=True)

    # validation the outputs of the flow or job
    assert len(responses) == 3
    for job, _parents in flow_or_job.iterflow():
        assert len(responses[job.uuid]) == 1


def test_relax_run_silicon_standard(mock_abinit, abinit_test_dir, clean_dir):
    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    # load the initial structure, the maker and the ref_paths from the test_dir
    test_dir = (
        abinit_test_dir / "flows" / "core" / "RelaxFlowMaker" / "silicon_standard"
    )
    structure = Structure.from_file(test_dir / "initial_structure.json.gz")
    maker_info = loadfn(test_dir / "maker.json.gz")
    maker = maker_info["maker"]
    ref_paths = loadfn(test_dir / "ref_paths.json.gz")

    mock_abinit(ref_paths)

    # make the flow or job, run it and ensure that it finished running successfully
    flow_or_job = maker.make(structure)
    responses = run_locally(flow_or_job, create_folders=True, ensure_success=True)

    # validation the outputs of the flow or job
    assert len(responses) == 2
    for job, _parents in flow_or_job.iterflow():
        assert len(responses[job.uuid]) == 1


def test_relax_ion_ioncell_relaxation():
    from atomate2.abinit.flows.core import RelaxFlowMaker

    settings = {"nband": 100}
    maker = RelaxFlowMaker.ion_ioncell_relaxation(user_abinit_settings=settings)
    assert len(maker.relaxation_makers) == 2
    assert (
        maker.relaxation_makers[0].input_set_generator.user_abinit_settings == settings
    )
    assert (
        maker.relaxation_makers[1].input_set_generator.user_abinit_settings == settings
    )
