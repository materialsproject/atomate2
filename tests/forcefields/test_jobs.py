from pytest import approx, importorskip

importorskip("quippy")


def test_chgnet_static_maker(si_structure):
    from jobflow import run_locally

    from atomate2.forcefields.jobs import CHGNetStaticMaker
    from atomate2.forcefields.schemas import ForceFieldTaskDocument

    task_doc_kwargs = {"ionic_step_data": ("structure", "energy")}

    # generate job
    job = CHGNetStaticMaker(task_document_kwargs=task_doc_kwargs).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.745277404785156, rel=1e-4)
    assert output1.output.ionic_steps[-1].magmoms is None
    assert output1.output.n_steps == 1


def test_chgnet_relax_maker(si_structure):
    from jobflow import run_locally

    from atomate2.forcefields.jobs import CHGNetRelaxMaker
    from atomate2.forcefields.schemas import ForceFieldTaskDocument

    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    # generate job
    job = CHGNetRelaxMaker(steps=25).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.74523544, rel=1e-4)
    assert output1.output.ionic_steps[-1].magmoms[0] == approx(0.00211287, rel=1e-4)
    assert output1.output.n_steps == 12


def test_m3gnet_static_maker(si_structure):
    from jobflow import run_locally

    from atomate2.forcefields.jobs import M3GNetStaticMaker
    from atomate2.forcefields.schemas import ForceFieldTaskDocument

    task_doc_kwargs = {"ionic_step_data": ("structure", "energy")}

    # generate job
    job = M3GNetStaticMaker(task_document_kwargs=task_doc_kwargs).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.71126747, rel=1e-4)
    assert output1.output.n_steps == 1


def test_m3gnet_relax_maker(si_structure):
    from jobflow import run_locally

    from atomate2.forcefields.jobs import M3GNetRelaxMaker
    from atomate2.forcefields.schemas import ForceFieldTaskDocument

    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    # generate job
    job = M3GNetRelaxMaker(steps=25).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.710836410522461, rel=1e-4)
    assert output1.output.n_steps == 14


def test_gap_static_maker(si_structure, test_dir):
    from jobflow import run_locally

    from atomate2.forcefields.jobs import GAPStaticMaker
    from atomate2.forcefields.schemas import ForceFieldTaskDocument

    task_doc_kwargs = {"ionic_step_data": ("structure", "energy")}

    # generate job
    # Test files have been provided by Yuanbin Liu (University of Oxford)
    job = GAPStaticMaker(
        potential_args_str="IP GAP",
        potential_param_file_name=str(
            test_dir / "forcefields" / "gap" / "gap_file.xml"
        ),
        task_document_kwargs=task_doc_kwargs,
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.8523, rel=1e-4)
    assert output1.output.n_steps == 1


def test_gap_relax_maker(si_structure, test_dir):
    from jobflow import run_locally

    from atomate2.forcefields.jobs import GAPRelaxMaker
    from atomate2.forcefields.schemas import ForceFieldTaskDocument

    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    # generate job
    # Test files have been provided by Yuanbin Liu (University of Oxford)
    job = GAPRelaxMaker(
        potential_param_file_name=test_dir / "forcefields" / "gap" / "gap_file.xml",
        steps=25,
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validating the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.8523, rel=1e-4)
    assert output1.output.n_steps == 17
