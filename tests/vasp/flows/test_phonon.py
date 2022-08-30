def test_primitive_different_kpath(mock_vasp, clean_dir, test_dir):
    # TODO: adapt
    ref_paths = {
        "non-scf uniform T=0.0": "Si_elph_renorm/non-scf_uniform_T=0.0",
        "non-scf uniform T=100.0": "Si_elph_renorm/non-scf_uniform_T=100.0",
        "non-scf uniform bulk supercell": "Si_elph_renorm/non-scf_uniform_bulk_supercell",
        "static": "Si_elph_renorm/static",
        "static T=0.0": "Si_elph_renorm/static_T=0.0",
        "static T=100.0": "Si_elph_renorm/static_T=100.0",
        "static bulk supercell": "Si_elph_renorm/static_bulk_supercell",
        "supercell electron phonon displacements": "Si_elph_renorm/supercell_electron_phonon_displacements",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "non-scf uniform T=0.0": {"incar_settings": ["NSW", "ISMEAR", "IBRION"]},
        "non-scf uniform T=100.0": {"incar_settings": ["NSW", "ISMEAR", "IBRION"]},
        "non-scf uniform bulk supercell": {
            "incar_settings": ["NSW", "ISMEAR", "IBRION"]
        },
        "static": {"incar_settings": ["NSW", "ISMEAR", "IBRION"]},
        "static T=0.0": {"incar_settings": ["NSW", "ISMEAR", "IBRION"]},
        "static T=100.0": {"incar_settings": ["NSW", "ISMEAR", "IBRION"]},
        "static bulk supercell": {"incar_settings": ["NSW", "ISMEAR", "IBRION"]},
        "supercell electron phonon displacements": {
            "incar_settings": ["NSW", "ISMEAR", "IBRION"]
        },
    }
    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)


def test_my_flow(mock_vasp, clean_dir, si_structure):
    from jobflow import run_locally

    # mapping from job name to directory containing test files
    ref_paths = {
        "dielectric": "phonon_flow/dielectric",
        "phonon static 1/8": "phonon_flow/phonon_static_1_8",
        "phonon static 2/8": "phonon_flow/phonon_static_2_8",
        "phonon static 3/8": "phonon_flow/phonon_static_3_8",
        "phonon static 4/8": "phonon_flow/phonon_static_4_8",
        "phonon static 5/8": "phonon_flow/phonon_static_5_8",
        "phonon static 6/8": "phonon_flow/phonon_static_6_8",
        "phonon static 7/8": "phonon_flow/phonon_static_7_8",
        "phonon static 8/8": "phonon_flow/phonon_static_8_8",
        "tight relax 1": "phonon_flow/tight_relax_1",
        "tight relax 2": "phonon_flow/tight_relax_2",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "dielectric": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 1/8": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 2/8": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 3/8": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 4/8": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 5/8": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 6/8": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 7/8": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 8/8": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # !!! Generate job
    job = MyMaker().make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # !!! validation on the outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == pytest.approx(-10.85037078)
