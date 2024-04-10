from pathlib import Path

import numpy as np
from emmet.core.classical_md.openmm import CalculationOutput


def test_calc_output_from_directory(output_dir):
    # Call the from_directory function
    calc_out = CalculationOutput.from_directory(
        output_dir,
        "state.csv",
        "trajectory.dcd",
        elapsed_time=10.0,
        n_steps=1500,
        state_interval=100,
    )

    # Assert the expected attributes of the CalculationOutput object
    assert isinstance(calc_out, CalculationOutput)
    assert calc_out.dir_name == str(output_dir)
    assert calc_out.elapsed_time == 10.0
    assert calc_out.traj_file == "trajectory.dcd"
    assert calc_out.state_file == "state.csv"

    # Assert the contents of the state data
    assert np.array_equal(calc_out.steps_reported[:3], [100, 200, 300])
    assert np.allclose(calc_out.potential_energy[:3], [-26192.4, -25648.6, -25149.6])
    assert np.allclose(calc_out.kinetic_energy[:3], [609.4, 1110.4, 1576.4], atol=0.1)
    assert np.allclose(calc_out.total_energy[:3], [-25583.1, -24538.1, -23573.2])
    assert np.allclose(calc_out.temperature[:3], [29.6, 54.0, 76.6], atol=0.1)
    assert np.allclose(calc_out.volume[:3], [21.9, 21.9, 21.9], atol=0.1)
    assert np.allclose(calc_out.density[:3], [1.0, 0.99, 0.99], atol=0.1)

    # Assert the existence of the DCD and state files
    assert Path(calc_out.dir_name, calc_out.traj_file).exists()
    assert Path(calc_out.dir_name, calc_out.state_file).exists()
