class TestStaticMaker:
    def test_run_silicon_standard(self, mock_abinit, abinit_test_dir, clean_dir):
        from jobflow import run_locally
        from monty.serialization import loadfn
        from pymatgen.core.structure import Structure

        from atomate2.abinit.schemas.core import AbinitTaskDocument

        # load the initial structure, the maker and the ref_paths from the test_dir
        test_dir = abinit_test_dir / "jobs" / "StaticMaker" / "silicon_standard"
        structure = Structure.from_file(test_dir / "initial_structure.json")
        maker_info = loadfn(test_dir / "maker.json")
        maker = maker_info["maker"]
        ref_paths = loadfn(test_dir / "ref_paths.json")

        mock_abinit(ref_paths)

        # make the job, run it and ensure that it finished running successfully
        job = maker.make(structure)
        responses = run_locally(job, create_folders=True, ensure_success=True)

        # validation the outputs of the job
        output1 = responses[job.uuid][1].output
        assert isinstance(output1, AbinitTaskDocument)
        assert output1.structure == structure
        assert output1.run_number == 1

    def test_run_silicon_restarts(self, mock_abinit, abinit_test_dir, clean_dir):
        from jobflow import run_locally
        from monty.serialization import loadfn
        from pymatgen.core.structure import Structure

        from atomate2.abinit.schemas.core import AbinitTaskDocument

        # load the initial structure, the maker and the ref_paths from the test_dir
        test_dir = abinit_test_dir / "jobs" / "StaticMaker" / "silicon_restarts"
        structure = Structure.from_file(test_dir / "initial_structure.json")
        maker_info = loadfn(test_dir / "maker.json")
        maker = maker_info["maker"]
        ref_paths = loadfn(test_dir / "ref_paths.json")

        mock_abinit(ref_paths)

        # make the job, run it and ensure that it finished running successfully
        job = maker.make(structure)
        responses = run_locally(job, create_folders=True, ensure_success=True)

        # validation the outputs of the job
        output1 = responses[job.uuid][1].output
        assert isinstance(output1, AbinitTaskDocument)
        assert output1.structure == structure
        assert output1.run_number == 1
        output2 = responses[job.uuid][2].output
        assert isinstance(output2, AbinitTaskDocument)
        assert output2.structure == structure
        assert output2.run_number == 2


class TestRelaxMaker:
    def test_run_silicon_scaled1p2_standard(
        self, mock_abinit, abinit_test_dir, clean_dir
    ):
        from jobflow import run_locally
        from monty.serialization import loadfn
        from pymatgen.core.structure import Structure

        from atomate2.abinit.schemas.core import AbinitTaskDocument

        # load the initial structure, the maker and the ref_paths from the test_dir
        test_dir = (
            abinit_test_dir / "jobs" / "RelaxMaker" / "silicon_scaled1p2_standard"
        )
        structure = Structure.from_file(test_dir / "initial_structure.json")
        maker_info = loadfn(test_dir / "maker.json")
        maker = maker_info["maker"]
        ref_paths = loadfn(test_dir / "ref_paths.json")

        mock_abinit(ref_paths)

        # make the flow or job, run it and ensure that it finished running successfully
        flow_or_job = maker.make(structure)
        responses = run_locally(flow_or_job, create_folders=True, ensure_success=True)

        # validation the outputs of the flow or job
        output1 = responses[flow_or_job.uuid][1].output
        assert isinstance(output1, AbinitTaskDocument)
        assert output1.run_number == 1
