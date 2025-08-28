from pathlib import Path

import pytest
from jobflow import run_locally
from monty.serialization import loadfn
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Xdatcar

from atomate2.forcefields.neb import ForceFieldNebFromImagesMaker


def test_neb_from_images(test_dir, clean_dir):
    endpoints = [
        Structure.from_file(
            test_dir
            / "vasp"
            / "Si_NEB"
            / f"relax_endpoint_{1 + i}"
            / "inputs"
            / "POSCAR.gz"
        )
        for i in range(2)
    ]

    images = endpoints[0].interpolate(endpoints[1], nimages=4, autosort_tol=0.5)

    job = ForceFieldNebFromImagesMaker(
        force_field_name="MATPES_PBE",
        traj_file="XDATCAR_si_self_diffusion",
        traj_file_fmt="xdatcar",
        relax_kwargs={"fmax": 0.5},
    ).make(images)

    response = run_locally(job)
    output = response[job.uuid][1].output

    cwd = next(Path(p) for p in output.tags if Path(p).exists())
    xdatcars = [
        Xdatcar(cwd / f"XDATCAR_si_self_diffusion-image-{i + 1}") for i in range(5)
    ]

    # Check that trajectory initial and final images are consistent with document
    assert all(
        xdatcars[i].structures[0] == image
        for i, image in enumerate(output.initial_images)
    )

    all(xdatcars[i].structures[-1] == image for i, image in enumerate(output.images))

    assert all(
        output.energies[i] == pytest.approx(energy)
        for i, energy in enumerate(
            [
                -339.3764953613281,
                -339.2301025390625,
                -338.865234375,
                -339.23004150390625,
                -339.37652587890625,
            ]
        )
    )

    assert output.state.value == "successful"
    assert "forces not converged" in output.tags

    images = endpoints[0].interpolate(endpoints[1], nimages=2, autosort_tol=0.5)
    job = ForceFieldNebFromImagesMaker(
        force_field_name="MACE",
        traj_file="si_self_diffusion.json.gz",
        traj_file_fmt="pmg",
        relax_kwargs={"fmax": 0.5},
    ).make(images)

    response = run_locally(job)
    output = response[job.uuid][1].output

    trajectories = [
        loadfn(cwd / f"si_self_diffusion-image-{idx + 1}.json.gz") for idx in range(3)
    ]

    assert all(
        trajectories[i].frame_properties[-1]["energy"] == pytest.approx(energy)
        for i, energy in enumerate(output.energies)
    )
