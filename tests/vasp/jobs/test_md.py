import pytest
from emmet.core.tasks import TaskDoc
from emmet.core.types.enums import VaspObject
from emmet.core.vasp.calculation import IonicStep
from jobflow import run_locally
from pymatgen.core import Element, Structure

from atomate2.vasp.jobs.md import MDMaker


def test_molecular_dynamics(mock_vasp, clean_dir, si_structure):
    # mapping from job name to directory containing test files
    ref_paths = {"molecular dynamics": "Si_molecular_dynamics/molecular_dynamics"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "molecular dynamics": {
            "incar_settings": [
                "IBRION",
                "TBEN",
                "TEND",
                "NSW",
                "POTIM",
                "MDALGO",
                "ISIF",
            ]
        }
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate job
    job = MDMaker().make(si_structure)
    nsw = 3
    job.maker.input_set_generator.user_incar_settings["NSW"] = nsw

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation on the output

    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert output1.output.energy == pytest.approx(-11.46520398)

    # check ionic steps stored as pymatgen Trajectory
    assert output1.calcs_reversed[0].output.ionic_steps is None
    traj = output1.vasp_objects[VaspObject.TRAJECTORY]
    assert all(
        len(getattr(traj, k)) == nsw for k in ("energy", "forces", "lattice", "stress")
    )
    # check that a frame property can be converted to an IonicStep
    energies = [-11.47041923, -11.46905352, -11.46520398]
    for idx in range(traj.num_ionic_steps):
        ionic_step = IonicStep(
            **{
                k: getattr(traj, k)[idx]
                for k in ("energy", "forces", "stress", "e_wo_entrp")
            },
            structure=Structure(
                traj.lattice[idx],
                [Element.from_Z(z) for z in traj.elements],
                traj.cart_coords[idx],
                coords_are_cartesian=True,
            ),
        )
        assert ionic_step.e_wo_entrp == pytest.approx(energies[idx])
