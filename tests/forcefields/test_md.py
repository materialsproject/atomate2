""" Tests for forcefield MD flows. """

import numpy as np
import pytest
import torch
from ase import units
from jobflow import run_locally
from monty.serialization import loadfn

from atomate2.forcefields.md import CHGNetMDMaker, M3GNetMDMaker, MACEMDMaker

_to_maker = {"CHGNet": CHGNetMDMaker, "M3GNet": M3GNetMDMaker, "MACE": MACEMDMaker}


@pytest.mark.parametrize("ff_name", ["CHGNet", "M3GNet", "MACE"])
def test_ml_ff_md_maker(ff_name, si_structure, clean_dir):
    nsteps = 5

    ref_energies_per_atom = {
        "CHGNet": -5.30686092376709,
        "M3GNet": -5.417105674743652,
        "MACE": -5.33246374130249,
    }

    structure = si_structure.to_conventional() * (2, 2, 2)

    # MACE changes the default dtype, ensure consistent dtype here
    torch.set_default_dtype(torch.float32)

    job = _to_maker[ff_name](nsteps=nsteps, traj_file="md_traj.json.gz").make(structure)
    response = run_locally(job, ensure_success=True)
    taskdoc = response[next(iter(response))][1].output

    # Check that energies are reasonably close to reference values
    assert taskdoc.output.energy / len(structure) == pytest.approx(
        ref_energies_per_atom[ff_name], abs=0.1
    )

    # Check that we have the right number of MD steps
    # ASE logs the initial structure energy, and doesn"t count this as an MD step
    assert taskdoc.output.ionic_steps[0].structure == structure
    assert len(taskdoc.output.ionic_steps) == nsteps + 1

    # Check that the ionic steps have the expected physical properties
    assert all(
        key in step.model_dump()
        for key in ("energy", "forces", "stress", "structure")
        for step in taskdoc.output.ionic_steps
    )

    # Check that the trajectory has expected physical properties
    assert taskdoc.included_objects == ["trajectory"]
    assert len(taskdoc.forcefield_objects["trajectory"]) == nsteps + 1
    assert all(
        key in step
        for key in ("energy", "forces", "stress", "velocities", "temperature")
        for step in taskdoc.forcefield_objects["trajectory"].frame_properties
    )

    # Check that traj file written to disk is consistent with trajectory
    # stored to the task document
    traj_from_file = loadfn("md_traj.json.gz")

    assert len(traj_from_file["energy"]) == nsteps + 1
    _traj_key_to_object_key = {
        "stresses": "stress",
    }
    assert all(
        np.all(
            traj_from_file[key][idx]
            == taskdoc.forcefield_objects["trajectory"].frame_properties[idx][
                _traj_key_to_object_key.get(key, key)
            ]
        )
        for key in ("energy", "temperature", "forces", "velocities")
        for idx in range(nsteps + 1)
    )

@pytest.mark.parametrize("ff_name", ["MACE", "CHGNet"])
def test_temp_schedule(ff_name, si_structure, clean_dir):
    nsteps = 200
    temp_schedule = [300, 3000]

    structure = si_structure.to_conventional() * (2, 2, 2)

    # MACE changes the default dtype, ensure consistent dtype here
    torch.set_default_dtype(torch.float32)

    job = _to_maker[ff_name](
        nsteps=nsteps, traj_file="md_traj.json.gz",
        dynamics="nose-hoover",
        temperature=temp_schedule,
        ase_md_kwargs=dict(ttime=50.0 * units.fs, pfactor=None)
    ).make(structure)
    response = run_locally(job, ensure_success=True)
    taskdoc = response[next(iter(response))][1].output


    temp_history = [
        step["temperature"] for step in taskdoc.forcefield_objects["trajectory"].frame_properties
    ]

    assert temp_history[-1] > temp_schedule[0]


@pytest.mark.parametrize("ff_name", ["MACE", "CHGNet"])
def test_press_schedule(ff_name, si_structure, clean_dir):
    nsteps = 200
    press_schedule = [0, 1e4]

    structure = si_structure.to_conventional() * (2, 2, 2)

    # MACE changes the default dtype, ensure consistent dtype here
    torch.set_default_dtype(torch.float32)

    job = _to_maker[ff_name](
        nsteps=nsteps, traj_file="md_traj.json.gz",
        dynamics="nose-hoover",
        pressure=press_schedule,
        ase_md_kwargs=dict(
            ttime=50.0 * units.fs,
            pfactor=(75.0 * units.fs)**2 * units.GPa,
            )
    ).make(structure)
    response = run_locally(job, ensure_success=True)
    taskdoc = response[next(iter(response))][1].output

    stress_history = [
        step["stresses"] for step in taskdoc.forcefield_objects["trajectory"].frame_properties
    ]

    assert stress_history[-1].trace() > stress_history[0].trace()
