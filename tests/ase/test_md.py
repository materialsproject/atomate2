"""
Test ASE MD implementation.

Most of the heavy tests of the AseMDMaker class are via forcefields
Light tests here to validate that base classes work as intended
"""

import os

import pytest
from jobflow import run_locally

from atomate2.ase.md import GFNxTBMDMaker, LennardJonesMDMaker
from atomate2.ase.schemas import AseStructureTaskDoc

try:
    from tblite.ase import TBLite
except ImportError:
    TBLite = None

name_to_maker = {"LJ": LennardJonesMDMaker}
if TBLite is not None:
    name_to_maker["GFN-xTB"] = GFNxTBMDMaker

_mb_velocity_seed = 2820285082114


@pytest.mark.parametrize("calculator_name", list(name_to_maker))
def test_ase_nvt_maker(calculator_name, lj_fcc_ne_pars, fcc_ne_structure, clean_dir):
    reference_energies = {
        "LJ": -0.0179726955438795,
        "GFN-xTB": -160.93692979071128,
    }

    md_job = name_to_maker[calculator_name](
        calculator_kwargs=lj_fcc_ne_pars if calculator_name == "LJ" else {},
        mb_velocity_seed=_mb_velocity_seed,
        temperature=1000,
        ensemble="nvt",
        n_steps=100,
        tags=["test"],
        store_trajectory="partial",
    ).make(fcc_ne_structure)

    response = run_locally(md_job)
    output = response[md_job.uuid][1].output

    assert isinstance(output, AseStructureTaskDoc)
    assert output.tags == ["test"]
    assert output.output.energy_per_atom == pytest.approx(
        reference_energies[calculator_name]
    )
    assert output.structure.volume == pytest.approx(fcc_ne_structure.volume)


@pytest.mark.parametrize("calculator_name", ["LJ"])
def test_ase_npt_maker(calculator_name, lj_fcc_ne_pars, fcc_ne_structure, tmp_dir):
    os.environ["OMP_NUM_THREADS"] = "1"

    reference_energies = {
        "LJ": 0.01705592581943574,
    }

    structure = fcc_ne_structure.to_conventional() * (3, 3, 3)

    md_job = name_to_maker[calculator_name](
        calculator_kwargs=lj_fcc_ne_pars if calculator_name == "LJ" else {},
        mb_velocity_seed=_mb_velocity_seed,
        temperature=1000,
        ensemble="npt",
        n_steps=(n_steps := 100),
        pressure=[0, 10],
        dynamics="nose-hoover",
        traj_file="XDATCAR",
        traj_file_fmt="xdatcar",
        store_trajectory="partial",
        ionic_step_data=("energy", "stress"),
    ).make(structure)

    response = run_locally(md_job)
    output = response[md_job.uuid][1].output

    assert isinstance(output, AseStructureTaskDoc)
    assert output.output.energy_per_atom == pytest.approx(
        reference_energies[calculator_name]
    )

    # TODO: improve XDATCAR parsing test when class is fixed in pmg
    assert os.path.isfile("XDATCAR")

    assert len(output.objects["trajectory"]) == n_steps
