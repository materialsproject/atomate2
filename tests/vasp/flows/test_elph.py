import pytest
from jobflow import run_locally

from atomate2.vasp.flows.elph import ElectronPhononMaker
from atomate2.vasp.schemas.elph import ElectronPhononRenormalisationDoc


def test_elph_renormalisation(mock_vasp, clean_dir, si_structure):
    # map job name to directory containing test files
    ref_paths = {
        "non-scf uniform T=0.0": "Si_elph_renorm/non-scf_uniform_T=0.0",
        "non-scf uniform T=100.0": "Si_elph_renorm/non-scf_uniform_T=100.0",
        "non-scf uniform bulk supercell": "Si_elph_renorm/"
        "non-scf_uniform_bulk_supercell",
        "static": "Si_elph_renorm/static",
        "static T=0.0": "Si_elph_renorm/static_T=0.0",
        "static T=100.0": "Si_elph_renorm/static_T=100.0",
        "static bulk supercell": "Si_elph_renorm/static_bulk_supercell",
        "supercell electron phonon displacements": "Si_elph_renorm/"
        "supercell_electron_phonon_displacements",
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

    # generate flow
    flow = ElectronPhononMaker(
        min_supercell_length=3, temperatures=(0, 100), relax_maker=None
    ).make(si_structure)
    set_op = {
        "_set": {"input_set_generator->user_kpoints_settings->reciprocal_density": 50}
    }
    flow.update_maker_kwargs(set_op, name_filter="static", dict_mod=True)
    flow.update_maker_kwargs(set_op, name_filter="non-scf", dict_mod=True)

    # run the flow and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validation on the outputs
    renorm_output = responses[flow.output.uuid][1].output
    assert isinstance(renorm_output, ElectronPhononRenormalisationDoc)
    assert renorm_output.delta_band_gaps == pytest.approx([-0.4889, -0.4885], rel=1e-3)
    assert renorm_output.chemsys == "Si"
