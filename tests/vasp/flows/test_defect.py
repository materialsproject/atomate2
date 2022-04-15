from atomate2.vasp.flows.defect import ConfigurationCoordinateMaker
from atomate2.vasp.schemas.defect import FiniteDifferenceDocument


def test_ccd_maker(mock_vasp, clean_dir, test_dir):
    from jobflow import run_locally
    from pymatgen.core import Structure

    from atomate2.vasp.flows.defect import ConfigurationCoordinateMaker
    from atomate2.vasp.schemas.defect import CCDDocument

    # mapping from job name to directory containing test files
    # mapping from job name to directory containing test files
    ref_paths = {
        "relax q1": "Si_config_coord/relax_q1",
        "relax q2": "Si_config_coord/relax_q2",
        "static q1 0": "Si_config_coord/static_q1_0",
        "static q1 1": "Si_config_coord/static_q1_1",
        "static q1 2": "Si_config_coord/static_q1_2",
        "static q1 3": "Si_config_coord/static_q1_3",
        "static q1 4": "Si_config_coord/static_q1_4",
        "static q2 0": "Si_config_coord/static_q2_0",
        "static q2 1": "Si_config_coord/static_q2_1",
        "static q2 2": "Si_config_coord/static_q2_2",
        "static q2 3": "Si_config_coord/static_q2_3",
        "static q2 4": "Si_config_coord/static_q2_4",
        "finite diff q1": "Si_config_coord/finite_diff_q1",
        "finite diff q2": "Si_config_coord/finite_diff_q2",
    }
    fake_run_vasp_kwargs = {k: {"incar_settings": ["ISIF"]} for k in ref_paths}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    si_defect = Structure.from_file(
        test_dir / "vasp" / "Si_config_coord" / "relax_q1" / "inputs" / "POSCAR"
    )

    # generate flow
    ccd_maker = ConfigurationCoordinateMaker(distortions=(-0.2, -0.1, 0, 0.1, 0.2))
    flow = ccd_maker.make(si_defect, charge_state1=0, charge_state2=1)

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


def test_nonrad_maker(mock_vasp, clean_dir, test_dir, monkeypatch):
    from jobflow import run_locally
    from pymatgen.core import Structure

    from atomate2.vasp.flows.defect import NonRadiativeMaker

    # mapping from job name to directory containing test files
    ref_paths = {
        "relax q1": "Si_config_coord/relax_q1",
        "relax q2": "Si_config_coord/relax_q2",
        "static q1 0": "Si_config_coord/static_q1_0",
        "static q1 1": "Si_config_coord/static_q1_1",
        "static q1 2": "Si_config_coord/static_q1_2",
        "static q1 3": "Si_config_coord/static_q1_3",
        "static q1 4": "Si_config_coord/static_q1_4",
        "static q2 0": "Si_config_coord/static_q2_0",
        "static q2 1": "Si_config_coord/static_q2_1",
        "static q2 2": "Si_config_coord/static_q2_2",
        "static q2 3": "Si_config_coord/static_q2_3",
        "static q2 4": "Si_config_coord/static_q2_4",
        "finite diff q1": "Si_config_coord/finite_diff_q1",
        "finite diff q2": "Si_config_coord/finite_diff_q2",
    }
    fake_run_vasp_kwargs = {k: {"incar_settings": ["ISIF"]} for k in ref_paths}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    si_defect = Structure.from_file(
        test_dir / "vasp" / "Si_config_coord" / "relax_q1" / "inputs" / "POSCAR"
    )

    ccd_maker = ConfigurationCoordinateMaker(distortions=(-0.2, -0.1, 0, 0.1, 0.2))
    non_rad_maker = NonRadiativeMaker(ccd_maker=ccd_maker)

    flow = non_rad_maker.make(si_defect, charge_state1=0, charge_state2=1)

    # run the flow and ensure that it finished running successfully
    responses = run_locally(
        flow,
        create_folders=True,
        ensure_success=True,
    )

    fdiff_doc1: FiniteDifferenceDocument = responses[flow.jobs[-2].uuid][1].output
    fdiff_doc2: FiniteDifferenceDocument = responses[flow.jobs[-1].uuid][1].output
    wswq1 = fdiff_doc1.wswq_documents[0].to_wswq()
    wswq2 = fdiff_doc2.wswq_documents[0].to_wswq()

    assert len(fdiff_doc1.wswq_documents) == 5
    assert len(fdiff_doc2.wswq_documents) == 5
    assert wswq1.data.shape == (2, 4, 18, 18)
    assert wswq2.data.shape == (2, 4, 18, 18)
