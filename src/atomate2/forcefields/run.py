"""ASE wrapper.

This module provides a wrapper around ASE to run ASE jobs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase import units
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from jobflow.utils import ValueEnum
from scipy.interpolate import interp1d

from atomate2 import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.md.md import MolecularDynamics
    from custodian.custodian import Validator
# TODO: add custodian support

class JobType(ValueEnum):
    """
    Type of ASE job.

    - ``DIRECT``: Run ASE without custodian.
    - ``NORMAL``: Normal custodian :obj:`.ASEJob`.
    """

    DIRECT = "direct"
    NORMAL = "normal"

def run_ase(
    job_type: JobType | str = JobType.NORMAL,
    ase_cmd: str = SETTINGS.ASE_CMD,
    max_errors: int = None, # TODO: add default custodian max_errors
    scratch_dir: str = SETTINGS.CUSTODIAN_SCRATCH_DIR,
    validators: Sequence[Validator] = None, # TODO: add validators
    ase_job_kwargs: dict[str, Any] = None,
    custodian_kwargs: dict[str, Any] = None,
) -> None:
    pass


def run_ase_md(
    atoms: Atoms,
    calculator: Calculator,
    input_set: str | dict[str, Any],
) -> None:
    """Run ASE molecular dynamics."""
    atoms.calc = calculator

    ensemble = input_set.pop("ensemble", "nvt").lower()
    dynamics = input_set.pop("dynamics", NPT)
    nsteps = input_set.pop("nsteps", 1000)
    timestep = input_set.pop("timestep", 2) * units.fs
    temperature = input_set.pop("temperature", 300)
    pressure = input_set.pop("pressure", 0) * 0.1 * units.GPa

    trajectory = input_set.pop("trajectory", "trajectory.traj")

    if isinstance(temperature, list):
        tschedule = np.interp(
            np.arange(nsteps),
            np.arange(len(temperature)),
            temperature
        )
    else:
        tschedule = np.full(nsteps, temperature)

    if isinstance(pressure, list):
        pschedule = np.interp(np.arange(nsteps), np.arange(len(pressure)), pressure)
    elif isinstance(pressure, np.ndarray):
        pschedule = interp1d(np.arange(nsteps), pressure, kind="linear")
    else:
        pschedule = np.full(nsteps, pressure)


    MaxwellBoltzmannDistribution(atoms, temperature_K=tschedule[0])
    Stationary(atoms)

    if ensemble == "nve":
        dyn = dynamics(atoms, timestep=timestep, trajectory=trajectory, **input_set)
    elif ensemble == "nvt":
        dyn = dynamics(
            atoms,
            timestep=timestep,
            temperature_K=tschedule[0],
            trajectory=trajectory,
            **input_set
        )
    elif ensemble == "npt":
        dyn = dynamics(
            atoms,
            timestep=timestep,
            temperature_K=tschedule[0],
            pressure=pschedule[0],
            trajectory=trajectory,
            **input_set,
        )
    else:
        raise ValueError(f"Unknown ensemble: {ensemble}")

    def callback(dyn: MolecularDynamics = dyn) -> None:
        if ensemble == "nve":
            return
        dyn.set_temperature(tschedule[dyn.nsteps])
        if ensemble == "npt":
            return
        dyn.set_stress(pschedule[dyn.nsteps])

    dyn.attach(callback, interval=1)
    dyn.run(nsteps)








