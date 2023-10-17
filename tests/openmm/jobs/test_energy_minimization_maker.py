def test_energy_minimization_maker(alchemy_input_set, job_store):
    from atomate2.openmm.jobs.energy_minimization_maker import EnergyMinimizationMaker
    from jobflow import run_locally

    energy_minimization_job_maker = EnergyMinimizationMaker(dcd_reporter_interval=1)
    energy_minimization_job = energy_minimization_job_maker.make(input_set=alchemy_input_set)
    run_locally(energy_minimization_job, store=job_store)
