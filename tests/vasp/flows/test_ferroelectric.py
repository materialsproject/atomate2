def test_my_flow(mock_vasp, clean_dir, test_dir):
    import numpy as np
    from jobflow import run_locally
    from pymatgen.core import Structure
    from atomate2.vasp.schemas.task import TaskDocument
    from atomate2.vasp.schemas.ferroelectric import PolarizationDocument
    from atomate2.vasp.flows.ferroelectric import FerroelectricMaker
    from atomate2.vasp.powerups import update_user_incar_settings

    # mapping from job name to directory containing test files
    ref_paths = {'polarization interpolation_0': 'KNbO3_ferroelectric/polarization_interpolation_0',
     'polarization nonpolar': 'KNbO3_ferroelectric/polarization_nonpolar',
     'polarization polar': 'KNbO3_ferroelectric/polarization_polar'}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {'polarization interpolation_0': {'incar_settings': ['NSW', 'ISMEAR','LCALCPOL']},
     'polarization nonpolar': {'incar_settings': ['NSW', 'ISMEAR','LCALCPOL']},
     'polarization polar': {'incar_settings': ['NSW', 'ISMEAR','LCALCPOL']}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # !!! Generate job
    st_p = Structure.from_file(
        test_dir / "vasp" / "KNbO3_ferroelectric/polarization_polar" / "inputs" / "POSCAR"
    )
    st_np = Structure.from_file(
        test_dir / "vasp" / "KNbO3_ferroelectric/polarization_nonpolar" / "inputs" / "POSCAR"
    )
    
    flow = FerroelectricMaker(relax=False,nimages=1).make(st_p,st_np)

    flow = update_user_incar_settings(flow, {"ENCUT": 400,"ISPIN":1})

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # !!! validation on the energy
    output1 = responses[flow.jobs[0].uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == -40.65766597

    # !!! validation on the polarization change 
    output1 = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(output1, PolarizationDocument)
    assert output1.polarization_change == [0.0, 0.0, 47.659737191658785]
    assert output1.polarization_change_norm == 47.659737191658785

