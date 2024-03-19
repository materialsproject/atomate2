def test_base_openmm_maker(alchemy_input_set, job_store, task_details, caplog):
    from atomate2.openmm.jobs.base_openmm_maker import BaseOpenMMMaker
    from atomate2.openmm.schemas.openmm_task_document import OpenMMTaskDocument
    from openmm.app.simulation import Simulation
    from tempfile import TemporaryDirectory
    from jobflow import run_locally

    base_job_maker = BaseOpenMMMaker(
        state_reporter_interval=0,
        dcd_reporter_interval=0,
    )

    base_job = base_job_maker.make(input_set=alchemy_input_set)

    with TemporaryDirectory() as temp_dir:

        # Validate _setup_base_openmm_task
        sim = base_job_maker._setup_base_openmm_task(
            input_set=alchemy_input_set,
            output_dir=temp_dir,
        )
        assert isinstance(sim, Simulation)

        # Validate _setup_base_openmm_task
        task_doc = base_job_maker._close_base_openmm_task(
            sim,
            input_set=alchemy_input_set,
            context=sim.context,
            task_details=task_details,
            output_dir=temp_dir,
        )
        assert isinstance(task_doc, OpenMMTaskDocument)

    # Validate raising of RuntimeError raised because of NotImplementedError
    try:
        run_locally(base_job, store=job_store, ensure_success=True)
        assert False
    except RuntimeError:
        assert True

    # Validate NotImplementedError in captured logs
    assert "NotImplementedError" in caplog.record_tuples[2][2]


def test_add_reporters():
    assert False


def test_resolve_attr():
    assert False


def test_create_integrator():
    assert False


def test_create_simulation():
    assert False


def test_update_interchange():
    assert False


def test_create_task_doc():
    assert False
