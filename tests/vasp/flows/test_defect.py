from atomate2.vasp.schemas.defect import FiniteDiffDocument


def test_ccd_maker(mock_vasp, clean_dir, test_dir):
    from jobflow import run_locally
    from pymatgen.core import Structure

    from atomate2.vasp.flows.defect import ConfigurationCoordinateMaker
    from atomate2.vasp.powerups import update_user_incar_settings
    from atomate2.vasp.schemas.defect import CCDDocument

    # mapping from job name to directory containing test files
    ref_paths = {
        "relax q=0": "Si_config_coord/relax_q=0",
        "relax q=1": "Si_config_coord/relax_q=1",
        "static q=0 0": "Si_config_coord/static_q=0_0",
        "static q=0 1": "Si_config_coord/static_q=0_1",
        "static q=0 2": "Si_config_coord/static_q=0_2",
        "static q=0 3": "Si_config_coord/static_q=0_3",
        "static q=0 4": "Si_config_coord/static_q=0_4",
        "static q=1 0": "Si_config_coord/static_q=1_0",
        "static q=1 1": "Si_config_coord/static_q=1_1",
        "static q=1 2": "Si_config_coord/static_q=1_2",
        "static q=1 3": "Si_config_coord/static_q=1_3",
        "static q=1 4": "Si_config_coord/static_q=1_4",
    }
    fake_run_vasp_kwargs = {k: {"incar_settings": ["ISIF"]} for k in ref_paths}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    si_defect = Structure.from_file(
        test_dir / "vasp" / "Si_config_coord" / "relax_q=0" / "inputs" / "POSCAR"
    )
    INCAR_UPDATES = {
        "KSPACING": 1,
    }

    def update_calc_settings(flow):
        flow = update_user_incar_settings(flow, incar_updates=INCAR_UPDATES)
        return flow

    # generate flow
    maker = ConfigurationCoordinateMaker(distortions=(-0.2, -0.1, 0, 0.1, 0.2))
    flow = maker.make(si_defect, charge_state1=0, charge_state2=1)
    flow = update_calc_settings(flow)

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


def test_nonrad_maker(mock_vasp, clean_dir, test_dir, monkeypatch):
    from jobflow import run_locally
    from pymatgen.core import Structure

    from atomate2.vasp.flows.defect import NonRadMaker
    from atomate2.vasp.powerups import update_user_incar_settings

    # mapping from job name to directory containing test files
    ref_paths = {
        "relax q=0": "Si_config_coord/relax_q=0",
        "relax q=1": "Si_config_coord/relax_q=1",
        "static q=0 0": "Si_config_coord/static_q=0_0",
        "static q=0 1": "Si_config_coord/static_q=0_1",
        "static q=0 2": "Si_config_coord/static_q=0_2",
        "static q=0 3": "Si_config_coord/static_q=0_3",
        "static q=0 4": "Si_config_coord/static_q=0_4",
        "static q=1 0": "Si_config_coord/static_q=1_0",
        "static q=1 1": "Si_config_coord/static_q=1_1",
        "static q=1 2": "Si_config_coord/static_q=1_2",
        "static q=1 3": "Si_config_coord/static_q=1_3",
        "static q=1 4": "Si_config_coord/static_q=1_4",
        "finite diff q=0": "Si_config_coord/finite_diff_q=0",
        "finite diff q=1": "Si_config_coord/finite_diff_q=1",
    }
    fake_run_vasp_kwargs = {k: {"incar_settings": ["ISIF"]} for k in ref_paths}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    si_defect = Structure.from_file(
        test_dir / "vasp" / "Si_config_coord" / "relax_q=0" / "inputs" / "POSCAR"
    )
    INCAR_UPDATES = {
        "KSPACING": 1,
    }

    def update_calc_settings(flow):
        flow = update_user_incar_settings(flow, incar_updates=INCAR_UPDATES)
        return flow

    # generate flow
    maker = NonRadMaker(distortions=(-0.2, -0.1, 0, 0.1, 0.2))
    flow = maker.make(si_defect, charge_state1=0, charge_state2=1)
    flow = update_calc_settings(flow)

    def dont_copy_files(self, *args, **kwargs):
        pass

    # run the flow and ensure that it finished running successfully
    responses = run_locally(
        flow,
        create_folders=True,
        ensure_success=True,
    )

    fdiff_doc1: FiniteDiffDocument = responses[flow.jobs[-1].uuid][1].output
    # fdiff_doc1: FiniteDiffDocument = responses[flow.jobs[-2].uuid][1].output

    assert len(fdiff_doc1.wswq_documents) == 5
