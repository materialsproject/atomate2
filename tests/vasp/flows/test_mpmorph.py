"""Test MPMorph VASP flows."""

import pytest

from atomate2.common.flows.amorphous import EquilibriumVolumeMaker
from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.sets.core import MDSetGenerator
from pymatgen.io.vasp import Kpoints
from jobflow import run_locally

from pymatgen.core import Structure


def test_equilibrium_volume_maker(mock_vasp, clean_dir, vasp_test_dir):

    ref_paths = {
        "Equilibrium Volume Maker molecular dynamics 1": "Si_mp_morph/Si_0.8",
        "Equilibrium Volume Maker molecular dynamics 2": "Si_mp_morph/Si_1.0",
        "Equilibrium Volume Maker molecular dynamics 3": "Si_mp_morph/Si_1.2",
    }

    mock_vasp(ref_paths)

    intial_structure = Structure.from_file(
        f"{vasp_test_dir}/Si_mp_morph/Si_1.0/inputs/POSCAR.gz"
    )
    temperature: int = 300
    end_temp: int = 300
    steps_convergence: int = 20

    gamma_point = Kpoints(
        comment="Gamma only",
        num_kpts=1,
        kpts=[[0, 0, 0]],
        kpts_weights=[1.0],
    )
    incar_settings = {
        "ISPIN": 1,  # Do not consider magnetism in AIMD simulations
        "LREAL": "Auto",  # Peform calculation in real space for AIMD due to large unit cell size
        "LAECHG": False,  # Don't need AECCAR for AIMD
        "EDIFFG": None,  # Does not apply to MD simulations, see: https://www.vasp.at/wiki/index.php/EDIFFG
        "GGA": "PS",  # Just let VASP decide based on POTCAR - the default, PS yields the error below
        "LPLANE": False,  # LPLANE is recommended to be False on Cray machines (https://www.vasp.at/wiki/index.php/LPLANE)
        "LDAUPRINT": 0,
    }

    aimd_equil_maker = MDMaker(
        input_set_generator=MDSetGenerator(
            ensemble="nvt",
            start_temp=temperature,
            end_temp=end_temp,
            nsteps=steps_convergence,
            time_step=2,
            # adapted from MPMorph settings
            user_incar_settings=incar_settings,
            user_kpoints_settings=gamma_point,
        )
    )

    flow = EquilibriumVolumeMaker(
        md_maker=aimd_equil_maker,
    ).make(structure=intial_structure)

    responses = run_locally(flow, create_folders=True, ensure_success=True)

    uuids = [uuid for uuid in responses]
    # print([responses[uuid][1].output for uuid in responses])
    print("-----STARTING PRINT-----")
    # print([uuid for uuid in responses])
    print(responses[uuids[0]][1].output)
    print("-----ENDING PRINT-----")
    # asserting False so that stdout is printed by pytest

    assert len(uuids) == 7
    assert responses[uuids[-5]][1].output.output.structure.volume == 82.59487098351644
    assert responses[uuids[-5]][1].output.output.energy == -13.44200043
    assert responses[uuids[-4]][1].output.output.structure.volume == 161.31810738968053
    assert responses[uuids[-4]][1].output.output.energy == -35.97470303
    assert responses[uuids[-3]][1].output.output.structure.volume == 278.7576895693679
    assert responses[uuids[-3]][1].output.output.energy == -32.48531985
    assert responses[uuids[-2]][1].output == {
        "relax": {
            "energy": [-13.44200043, -35.97470303, -32.48531985],
            "volume": [82.59487098351644, 161.31810738968053, 278.7576895693679],
            "stress": [
                [
                    [2026.77697447, -180.19246839, -207.37762676],
                    [-180.1924744, 1441.18625768, 27.8884401],
                    [-207.37763076, 27.8884403, 1899.32511191],
                ],
                [
                    [36.98140157, 35.7070696, 76.84918574],
                    [35.70706985, 42.81953297, -25.20830843],
                    [76.84918627, -25.20830828, 10.40947728],
                ],
                [
                    [-71.39703139, -5.93838689, -5.13934938],
                    [-5.93838689, -72.87372166, -3.0206588],
                    [-5.13934928, -3.0206586, -57.65692738],
                ],
            ],
            "pressure": [1789.0961146866666, 30.070137273333334, -67.30922681],
            "EOS": {
                "b0": 424.97840444799994,
                "b1": 4.686114896027117,
                "v0": 171.51226566279973,
            },
        },
        "V0": 171.51226566279973,
        "Vmax": 278.7576895693679,
        "Vmin": 82.59487098351644,
    }
