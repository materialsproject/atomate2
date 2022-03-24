from atomate2.vasp.schemas.defect import CCDDocument


def test_ccd_maker(mock_vasp, clean_dir, test_dir):
    from jobflow import run_locally
    from pymatgen.core import Structure

    from atomate2.vasp.flows.defect import ConfigurationCoordinateMaker

    # mapping from job name to directory containing test files
    ref_paths = {
        "relax q=0": "Si_CCD/relax_q=0",
        "relax q=1": "Si_CCD/relax_q=1",
        "static q=0 0": "Si_CCD/static_q=0_0",
        "static q=0 1": "Si_CCD/static_q=0_1",
        "static q=0 2": "Si_CCD/static_q=0_2",
        "static q=0 3": "Si_CCD/static_q=0_3",
        "static q=0 4": "Si_CCD/static_q=0_4",
        "static q=1 0": "Si_CCD/static_q=1_0",
        "static q=1 1": "Si_CCD/static_q=1_1",
        "static q=1 2": "Si_CCD/static_q=1_2",
        "static q=1 3": "Si_CCD/static_q=1_3",
        "static q=1 4": "Si_CCD/static_q=1_4",
    }
    fake_run_vasp_kwargs = {k: {"incar_settings": ["ISIF"]} for k in ref_paths}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    si_defect = Structure.from_file(
        test_dir / "vasp" / "Si_CCD" / "relax_q=0" / "inputs" / "POSCAR"
    )

    # generate flow
    maker = ConfigurationCoordinateMaker(distortions=(-0.2, -0.1, 0, 0.1, 0.2))
    flow = maker.make(si_defect, charge_state1=0, charge_state2=1)

    # run the flow and ensure that it finished running successfully
    responses = run_locally(
        flow,
        create_folders=True,
        ensure_success=True,
    )

    ccd: CCDDocument = responses[flow.jobs[-1].uuid][1].output

    assert len(ccd.energies1) == 5
    assert len(ccd.energies2) == 5
    assert len(ccd.distortions1) == 5
    assert len(ccd.distortions2) == 5
    assert sum(len(row) for row in ccd.distorted_calcs_dirs) == 10
