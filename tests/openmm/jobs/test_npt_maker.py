def test_npt_maker(alchemy_input_set, job_store):
    from atomate2.openmm.jobs.energy_minimization_maker import EnergyMinimizationMaker
    from atomate2.openmm.schemas.openmm_task_document import OpenMMTaskDocument
    from atomate2.openmm.jobs.npt_maker import NPTMaker
    from jobflow import Flow, run_locally

    energy_minimization_job_maker = EnergyMinimizationMaker()
    npt_job_maker = NPTMaker(
        steps=1000,
        dcd_reporter_interval=10,
        state_reporter_interval=10,
    )
    energy_minimization_job = energy_minimization_job_maker.make(input_set=alchemy_input_set)
    npt_job = npt_job_maker.make(input_set=energy_minimization_job.output.calculation_output.input_set)
    flow = Flow(
        jobs=[
            energy_minimization_job,
            npt_job,
        ]
    )

    response = run_locally(flow=flow, store=job_store, ensure_success=True)
    output = response[npt_job.uuid][1].output
    assert isinstance(output, OpenMMTaskDocument)

    with job_store.docs_store as s:
        doc = s.query_one({"uuid": npt_job.uuid})
        assert doc["output"]["@class"] == "OpenMMTaskDocument"