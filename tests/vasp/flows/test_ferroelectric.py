import pytest


def test_my_flow(mock_vasp, clean_dir, test_dir):
    from emmet.core.tasks import TaskDoc
    from jobflow import run_locally
    from pymatgen.core import Structure

    from atomate2.vasp.flows.ferroelectric import FerroelectricMaker
    from atomate2.vasp.powerups import update_user_incar_settings
    from atomate2.vasp.schemas.ferroelectric import PolarizationDocument

    # mapping from job name to directory containing test files
    ref_paths = {
        "polarization interpolation_0": "KNbO3_ferroelectric/polarization_interpolation_0",  # noqa: E501
        "polarization nonpolar": "KNbO3_ferroelectric/polarization_nonpolar",
        "polarization polar": "KNbO3_ferroelectric/polarization_polar",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "polarization interpolation_0": {
            "incar_settings": ["NSW", "ISMEAR", "LCALCPOL"]
        },
        "polarization nonpolar": {"incar_settings": ["NSW", "ISMEAR", "LCALCPOL"]},
        "polarization polar": {"incar_settings": ["NSW", "ISMEAR", "LCALCPOL"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # !!! Generate job
    st_p = Structure.from_file(
        test_dir
        / "vasp"
        / "KNbO3_ferroelectric/polarization_polar"
        / "inputs"
        / "POSCAR"
    )
    st_np = Structure.from_file(
        test_dir
        / "vasp"
        / "KNbO3_ferroelectric/polarization_nonpolar"
        / "inputs"
        / "POSCAR"
    )
    st_interp = Structure.from_file(
        test_dir
        / "vasp"
        / "KNbO3_ferroelectric/polarization_interpolation_0"
        / "inputs"
        / "POSCAR"
    )

    flow = FerroelectricMaker(relax_maker=False, nimages=1).make(st_p, st_np)

    flow = update_user_incar_settings(flow, {"ENCUT": 400, "ISPIN": 1})

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # !!! validation on the energy
    output1 = responses[flow.jobs[0].uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert output1.output.energy == -40.65768215

    # !!! validation on the polarization change
    output1 = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(output1, PolarizationDocument)
    assert output1.polarization_change == pytest.approx([0.0, 0.0, 47.34548], rel=1e-6)
    assert output1.polarization_change_norm == pytest.approx(47.34548, rel=1e-6)

    # validate total number of structures
    assert len(output1.structures) == 3

    # validate interpolation made via pmg
    abc_1 = output1.structures[1].lattice.abc
    abc_p = output1.structures[0].lattice.abc
    abc_np = output1.structures[-1].lattice.abc
    assert abc_1 == st_interp.lattice.abc
    assert abc_p == st_p.lattice.abc
    assert abc_np == st_np.lattice.abc
