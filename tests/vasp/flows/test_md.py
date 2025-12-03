from emmet.core.types.enums import VaspObject
from jobflow import Flow

from atomate2 import SETTINGS
from atomate2.vasp.flows.md import MultiMDMaker
from atomate2.vasp.powerups import update_user_kpoints_settings
from atomate2.vasp.schemas.md import MultiMDOutput

if SETTINGS.VASP_USE_EMMET_MODELS:
    from emmet.core.trajectory import RelaxTrajectory

    TRAJ_TYPE = RelaxTrajectory
else:
    from pymatgen.core.trajectory import Trajectory

    TRAJ_TYPE = Trajectory


def test_multi_md_flow(mock_vasp, clean_dir, si_structure):
    from emmet.core.tasks import TaskDoc
    from jobflow import run_locally

    # mapping from job name to directory containing test files
    ref_paths = {
        "molecular dynamics 1": "Si_multi_md/molecular_dynamics_1",
        "molecular dynamics 2": "Si_multi_md/molecular_dynamics_2",
        "molecular dynamics 3": "Si_multi_md/molecular_dynamics_3",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "molecular dynamics 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "molecular dynamics 2": {"incar_settings": ["NSW", "ISMEAR"]},
        "molecular dynamics 3": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    mdflow1 = MultiMDMaker.from_parameters(
        nsteps=3, time_step=1, n_runs=2, ensemble="nvt", start_temp=300
    ).make(si_structure)
    mdflow2 = MultiMDMaker.from_parameters(
        nsteps=3, time_step=1, n_runs=1, ensemble="nvt", start_temp=300
    ).restart_from_uuid(mdflow1.jobs[-1].output)
    # set the name of the continuation MD Job, otherwise the folders for
    # files will conflict
    mdflow2.jobs[0].name = "molecular dynamics 3"

    flow = Flow([mdflow1, mdflow2])
    flow = update_user_kpoints_settings(
        flow, {"grid_density": 100}, name_filter="molecular dynamics"
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate the outputs
    output_md_1 = responses[flow.jobs[0].jobs[0].uuid][1].output
    traj = output_md_1.vasp_objects[VaspObject.TRAJECTORY]
    assert isinstance(traj, TRAJ_TYPE)
    if SETTINGS.VASP_USE_EMMET_MODELS:
        assert traj.num_ionic_steps == 3
        assert all(
            len(getattr(traj, k)) == traj.num_ionic_steps
            for k in ("energy", "forces", "lattice", "stress")
        )
    else:
        assert len(traj.frame_properties) == 3
    assert isinstance(output_md_1, TaskDoc)

    output_recap_1 = responses[flow.jobs[0].jobs[2].uuid][1].output
    assert len(output_recap_1.traj_ids) == 2
    assert len(output_recap_1.full_traj_ids) == 2
    assert isinstance(output_recap_1, MultiMDOutput)

    output_recap_2 = responses[flow.jobs[1].jobs[1].uuid][1].output
    assert len(output_recap_2.traj_ids) == 1
    assert len(output_recap_2.full_traj_ids) == 3
    assert isinstance(output_recap_1, MultiMDOutput)


def test_multi_md_flow_restart_from_uuid():
    # check that the correct reference is used if a string is passed
    ref_id = "475bf8ab-06ec-4222-8bad-6f9f3979f2ea"
    flow = MultiMDMaker().restart_from_uuid(ref_id)

    assert flow.jobs[0].function_kwargs["prev_dir"].uuid == ref_id
