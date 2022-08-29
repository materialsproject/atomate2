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
