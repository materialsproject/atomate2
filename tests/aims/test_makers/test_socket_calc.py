import os

import pytest
from pymatgen.core import Lattice

cwd = os.getcwd()


@pytest.mark.skip(reason="Currently not mocked and needs FHI-aims binary")
def test_static_socket_maker(si, species_dir, tmp_path):
    from jobflow import run_locally
    from pymatgen.io.aims.sets.core import SocketIOSetGenerator

    from atomate2.aims.jobs.core import SocketIOStaticMaker
    from atomate2.aims.schemas.task import AimsTaskDoc

    atoms = si
    atoms_list = [atoms, atoms.copy(), atoms.copy()]
    atoms_list[1].cart_coords[0, 0] += 0.02
    atoms_list[2].lattice = Lattice(atoms_list[2].lattice.matrix * 1.02)

    parameters = {
        "k_grid": [2, 2, 2],
        "species_dir": (species_dir / "light").as_posix(),
    }
    # generate job
    maker = SocketIOStaticMaker(
        input_set_generator=SocketIOSetGenerator(user_params=parameters)
    )
    maker.name = "socket"
    job = maker.make(atoms_list)

    # run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(cwd)

    # validation the outputs of the job
    outputs = responses[job.uuid][1].output
    assert isinstance(outputs, AimsTaskDoc)
    assert len(outputs.output.trajectory) == 3
    assert outputs.output.trajectory[0].properties["energy"] == pytest.approx(
        -15800.0997410132
    )
    assert outputs.output.trajectory[1].properties["energy"] == pytest.approx(
        -15800.0962356206
    )
    assert outputs.output.trajectory[2].properties["energy"] == pytest.approx(
        -15800.2028334278
    )
    # assert output1.output.energy == pytest.approx(-15800.099740991)
