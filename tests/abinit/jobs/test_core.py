class TestStaticMaker:
    def test_init(self, mock_abinit, clean_dir, si_structure):
        from jobflow import run_locally

        from atomate2.abinit.jobs.core import StaticMaker
        from atomate2.abinit.schemas.core import AbinitTaskDocument

        # mapping from job name to directory containing test files
        ref_paths = {"scf": "jobs/StaticMaker/Si"}

        mock_abinit(ref_paths)

        # generate job
        maker = StaticMaker.from_params(
            name="scf", ecut=4.0, kppa=10, spin_mode="unpolarized", nband=4
        )
        job = maker.make(si_structure)

        # run the flow or job and ensure that it finished running successfully
        responses = run_locally(job, create_folders=True, ensure_success=True)

        # validation the outputs of the job
        output1 = responses[job.uuid][1].output
        assert isinstance(output1, AbinitTaskDocument)
        assert output1.structure == si_structure
        assert output1.run_number == 1
