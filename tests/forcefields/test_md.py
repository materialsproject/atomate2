"""Tests for forcefield MD flows."""

import numpy as np
import pytest
import torch
from ase import units
from ase.io import Trajectory
from jobflow import run_locally
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher

from atomate2.forcefields.md import CHGNetMDMaker, M3GNetMDMaker, MACEMDMaker

_to_maker = {"CHGNet": CHGNetMDMaker, "M3GNet": M3GNetMDMaker, "MACE": MACEMDMaker}

# MACE changes the default dtype, ensure consistent dtype here
torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize("ff_name", ["CHGNet", "M3GNet", "MACE"])
def test_ml_ff_md_maker(ff_name, si_structure, clean_dir):
    nsteps = 5

    ref_energies_per_atom = {
        "CHGNet": -5.280157089233398,
        "M3GNet": -5.387282371520996,
        "MACE": -5.311369895935059,
    }

    structure = si_structure.to_conventional() * (2, 2, 2)
    # ASE can slightly change tolerances on structure positions
    matcher = StructureMatcher()

    job = _to_maker[ff_name](
        nsteps=nsteps,
        traj_file="md_traj.json.gz",
        traj_file_fmt="pmg",
        task_document_kwargs={"store_trajectory": "partial"},
    ).make(structure)
    response = run_locally(job, ensure_success=True)
    taskdoc = response[next(iter(response))][1].output

    # Check that energies are reasonably close to reference values
    assert taskdoc.output.energy / len(structure) == pytest.approx(
        ref_energies_per_atom[ff_name], abs=0.1
    )

    # Check that we have the right number of MD steps
    # ASE logs the initial structure energy, and doesn't count this as an MD step
    assert matcher.fit(taskdoc.output.ionic_steps[0].structure, structure)
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


@pytest.mark.parametrize("traj_file", ["trajectory.json.gz", "atoms.traj"])
def test_traj_file(traj_file, si_structure, clean_dir, ff_name="CHGNet"):
    nsteps = 5

    # Check that traj file written to disk is consistent with trajectory
    # stored to the task document

    if ".json.gz" in traj_file:
        traj_file_fmt = "pmg"
        traj_file_loader = loadfn
    else:
        traj_file_fmt = "ase"
        traj_file_loader = Trajectory

    structure = si_structure.to_conventional() * (2, 2, 2)
    job = _to_maker[ff_name](
        nsteps=nsteps,
        traj_file=traj_file,
        traj_file_fmt=traj_file_fmt,
    ).make(structure)
    response = run_locally(job, ensure_success=True)
    taskdoc = response[next(iter(response))][1].output

    traj_from_file = traj_file_loader(traj_file)

    assert len(traj_from_file) == nsteps + 1

    if traj_file_fmt == "pmg":
        assert all(
            np.all(
                traj_from_file.frame_properties[idx][key]
                == taskdoc.forcefield_objects["trajectory"]
                .frame_properties[idx]
                .get(key)
            )
            for key in ("energy", "temperature", "forces", "velocities")
            for idx in range(nsteps + 1)
        )
    elif traj_file_fmt == "ase":
        traj_as_dict = [
            {
                "energy": traj_from_file[idx].get_potential_energy(),
                "temperature": traj_from_file[idx].get_temperature(),
                "forces": traj_from_file[idx].get_forces(),
                "velocities": traj_from_file[idx].get_velocities(),
            }
            for idx in range(1, nsteps + 1)
        ]
        assert all(
            np.all(
                traj_as_dict[idx - 1][key]
                == taskdoc.forcefield_objects["trajectory"]
                .frame_properties[idx]
                .get(key)
            )
            for key in ("energy", "temperature", "forces", "velocities")
            for idx in range(1, nsteps + 1)
        )


@pytest.mark.parametrize("ff_name", ["MACE", "CHGNet"])
def test_temp_schedule(ff_name, si_structure, clean_dir):
    nsteps = 100
    temp_schedule = [300, 3000]

    structure = si_structure.to_conventional() * (2, 2, 2)

    job = _to_maker[ff_name](
        nsteps=nsteps,
        traj_file=None,
        dynamics="nose-hoover",
        temperature=temp_schedule,
        ase_md_kwargs=dict(ttime=50.0 * units.fs, pfactor=None),
    ).make(structure)
    response = run_locally(job, ensure_success=True)
    taskdoc = response[next(iter(response))][1].output

    temp_history = [
        step["temperature"]
        for step in taskdoc.forcefield_objects["trajectory"].frame_properties
    ]

    assert temp_history[-1] > temp_schedule[0]


@pytest.mark.parametrize("ff_name", ["MACE", "CHGNet"])
def test_press_schedule(ff_name, si_structure, clean_dir):
    nsteps = 100
    press_schedule = [0, 10]  # kbar

    structure = si_structure.to_conventional() * (3, 3, 3)

    job = _to_maker[ff_name](
        ensemble="npt",
        nsteps=nsteps,
        traj_file="md_traj.json.gz",
        traj_file_fmt="pmg",
        dynamics="nose-hoover",
        pressure=press_schedule,
        ase_md_kwargs=dict(
            ttime=50.0 * units.fs,
            pfactor=(75.0 * units.fs) ** 2 * units.GPa,
        ),
    ).make(structure)
    run_locally(job, ensure_success=True)
    # taskdoc = response[next(iter(response))][1].output

    traj_from_file = loadfn("md_traj.json.gz")

    stress_history = [
        sum(traj_from_file.frame_properties[idx]["stress"][:3]) / 3.0
        for idx in range(len(traj_from_file))
    ]

    assert stress_history[-1] < stress_history[0]
