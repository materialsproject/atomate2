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
    assert ccd_maker.distortions == (-0.2, -0.1, 0, 0.1, 0.2)
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
    assert ccd.relaxed_index1 == 2
    assert ccd.relaxed_index2 == 2


def test_nonrad_maker(mock_vasp, clean_dir, test_dir, monkeypatch):
    import numpy as np
    from jobflow import JobStore, run_locally
    from maggma.stores.mongolike import MemoryStore
    from pymatgen.core import Structure
    from pymatgen.io.vasp.outputs import WSWQ

    from atomate2.vasp.flows.defect import (
        ConfigurationCoordinateMaker,
        NonRadiativeMaker,
    )
    from atomate2.vasp.schemas.defect import FiniteDifferenceDocument

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
    docs_store = MemoryStore()
    data_store = MemoryStore()
    store = JobStore(docs_store, additional_stores={"data": data_store})
    responses = run_locally(
        flow,
        create_folders=True,
        ensure_success=True,
        store=store,
    )

    fdiff_doc1: FiniteDifferenceDocument = responses[flow.jobs[-2].uuid][1].output
    fdiff_doc2: FiniteDifferenceDocument = responses[flow.jobs[-1].uuid][1].output
    wswq1 = fdiff_doc1.wswqs[0]
    wswq2 = fdiff_doc2.wswqs[0]

    assert len(fdiff_doc1.wswqs) == 5
    assert len(fdiff_doc2.wswqs) == 5
    assert wswq1.me_real.shape == (2, 4, 18, 18)
    assert wswq2.me_imag.shape == (2, 4, 18, 18)
    for q in store.additional_stores["data"].query(
        {"job_uuid": {"$in": [flow.jobs[-2].uuid, flow.jobs[-1].uuid]}}
    ):
        assert q["data"] is not None
        wswq_p = WSWQ.from_dict(q["data"])
        wswq_p.me_real = np.array(wswq_p.me_real)
        wswq_p.me_imag = np.array(wswq_p.me_imag)
        assert wswq_p.me_real.shape == (2, 4, 18, 18)
        assert wswq_p.me_imag.shape == (2, 4, 18, 18)


def test_formation_energy_maker(mock_vasp, clean_dir, test_dir):
    from jobflow import JobStore, run_locally
    from maggma.stores.mongolike import MemoryStore
    from pymatgen.analysis.defect.core import Defect
    from pymatgen.analysis.defect.generators import SubstitutionGenerator
    from pymatgen.core import Structure
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    from atomate2.vasp.flows.defect import FormationEnergyMaker
    from atomate2.vasp.powerups import (
        update_user_incar_settings,
        update_user_kpoints_settings,
    )

    # mapping from job name to directory containing test files
    ref_paths = {
        "bulk relax": "GaN_Mg_defect/bulk_relax",
        "relax Mg_Ga-0 q=-2": "GaN_Mg_defect/Mg_Ga_0_q=-2",
        "relax Mg_Ga-0 q=-1": "GaN_Mg_defect/Mg_Ga_0_q=-1",
        "relax Mg_Ga-0 q=0": "GaN_Mg_defect/Mg_Ga_0_q=0",
        "relax Mg_Ga-0 q=1": "GaN_Mg_defect/Mg_Ga_0_q=1",
    }
    fake_run_vasp_kwargs = {k: {"incar_settings": ["ISIF"]} for k in ref_paths}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    struct_GaN = Structure.from_file(test_dir / "structures" / "GaN.cif")
    sub_gen = SubstitutionGenerator(structure=struct_GaN, substitutions={"Ga": ["Mg"]})

    maker = FormationEnergyMaker()
    flow = maker.make(sub_gen, sc_mat=[[2, 2, 0], [2, -2, 0], [0, 0, 1]])
    flow = update_user_kpoints_settings(flow, {"reciprocal_density": 64})
    flow = update_user_incar_settings(
        flow,
        {
            "LREAL": "Auto",
            "ALGO": "Normal",
            "ENCUT": 500,
            "GGA": None,
            "NELMIN": 6,
            "NCORE": 4,
            "EDIFFG": -0.1,
        },
    )

    # run the flow and ensure that it finished running successfully
    docs_store = MemoryStore()
    data_store = MemoryStore()
    store = JobStore(docs_store, additional_stores={"data": data_store})
    responses = run_locally(
        flow,
        create_folders=True,
        ensure_success=True,
        store=store,
    )

    results = responses[flow.jobs[-1].uuid][1].output
    res_defect = results["defect_calcs"]["Mg_Ga-0"]["defect"]
    assert isinstance(res_defect, Defect)
    for q, r_dict in results["defect_calcs"]["Mg_Ga-0"]["results"].items():
        assert isinstance(r_dict["entry"], ComputedStructureEntry)
