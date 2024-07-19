"""Test FHI-aims Equation of State workflow"""

import os

import pytest
from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.aims.flows.eos import AimsEosMaker
from atomate2.aims.jobs.core import RelaxMaker

cwd = os.getcwd()

# mapping from job name to directory containing test files
ref_paths = {
    "Relaxation calculation 1": "double-relax-si/relax-1",
    "Relaxation calculation 2": "double-relax-si/relax-2",
    "Relaxation calculation (fixed cell) deformation 0": "eos-si/0",
    "Relaxation calculation (fixed cell) deformation 1": "eos-si/1",
    "Relaxation calculation (fixed cell) deformation 2": "eos-si/2",
    "Relaxation calculation (fixed cell) deformation 3": "eos-si/3",
}


def test_eos(mock_aims, tmp_path, species_dir):
    """A test for the equation of state flow"""

    # a relaxed structure for the test
    a = 2.80791457
    si = Structure(
        lattice=[[0.0, a, a], [a, 0.0, a], [a, a, 0.0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # settings passed to fake_run_aims
    fake_run_kwargs = {}

    # automatically use fake AIMS
    mock_aims(ref_paths, fake_run_kwargs)

    # generate flow
    eos_relax_maker = RelaxMaker.fixed_cell_relaxation(
        user_params={
            "species_dir": (species_dir / "light").as_posix(),
            # "species_dir": "light",
            "k_grid": [2, 2, 2],
        }
    )

    flow = AimsEosMaker(
        initial_relax_maker=None, eos_relax_maker=eos_relax_maker, number_of_frames=4
    ).make(si)

    # Run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    os.chdir(cwd)

    output = responses[flow.jobs[-1].uuid][1].output
    assert "EOS" in output["relax"]
    # there is no initial calculation; fit using 4 points
    assert len(output["relax"]["energy"]) == 4
    assert output["relax"]["EOS"]["birch_murnaghan"]["b0"] == pytest.approx(
        0.4897486348366812
    )


def test_eos_from_parameters(mock_aims, tmp_path, si, species_dir):
    """A test for the equation of state flow, created from the common parameters"""

    # settings passed to fake_run_aims
    fake_run_kwargs = {}

    # automatically use fake AIMS
    mock_aims(ref_paths, fake_run_kwargs)

    # generate flow
    flow = AimsEosMaker.from_parameters(
        parameters={
            # TODO: to be changed after pymatgen PR is merged
            "species_dir": {
                "initial": species_dir,
                "eos": (species_dir / "light").as_posix(),
            },
            # "species_dir": "light",
            "k_grid": [2, 2, 2],
        },
        number_of_frames=4,
    ).make(si)

    # Run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    os.chdir(cwd)

    output = responses[flow.jobs[-1].uuid][1].output
    assert "EOS" in output["relax"]
    # there is an initial calculation; fit using 5 points
    assert len(output["relax"]["energy"]) == 5
    # the initial calculation also participates in the fit here
    assert output["relax"]["EOS"]["birch_murnaghan"]["b0"] == pytest.approx(
        0.5189578108402951
    )
