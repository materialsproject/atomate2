from pytest import approx


def test_static_maker(si_structure):
    from jobflow import run_locally

    from atomate2.forcefields.jobs import CHGNetStaticMaker
    from atomate2.forcefields.schemas import ForceFieldTaskDocument

    task_doc_kwargs = {"ionic_step_data": ("structure", "energy")}

    # generate job
    job = CHGNetStaticMaker(task_document_kwargs=task_doc_kwargs).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.745277404785156)
    assert output1.output.ionic_steps[-1].magmoms is None
    assert output1.output.n_steps == 1


def test_relax_maker(si_structure):
    from jobflow import run_locally

    from atomate2.forcefields.jobs import CHGNetRelaxMaker
    from atomate2.forcefields.schemas import ForceFieldTaskDocument

    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    # generate job
    job = CHGNetRelaxMaker(steps=25).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validating the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.745235443115234)
    assert output1.output.ionic_steps[-1].magmoms[0] == approx(0.002112872898578644)
    assert output1.output.n_steps == 12
