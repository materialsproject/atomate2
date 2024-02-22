"""Module defining core ASE MD input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pymatgen.io.core import InputGenerator

logger = logging.getLogger(__name__)

@dataclass
class MDSetGenerator(InputGenerator):
    """
    Class to generate ASE molecular dynamics input sets.

    Parameters
    ----------
    ensemble
        Molecular dynamics ensemble to run. Options include `nvt`, `nve`, and `npt`.
    nsteps
        Number of ionic steps for simulations.
    timestep
        The time step (in femtosecond) for the simulation.
    temp_schedule
        Temperature schedule for the simulation in Kelvin. Default is None. If None,
        the temperature is constant at the value of 300 K.
    press_schedule
        Pressure schedule for the simulation in kilobar. Default is None. If None,
        the pressure is constant at the value of 0. Positive values are for compression.
    **kwargs
        Other parameters passed to the InputGenerator.
    """

    ensemble: str = "nvt"
    nsteps: int = 1000
    timestep: float = 2
    temp_schedule: list[float] | None = None
    press_schedule: list[float] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    # TODO: revise for ASE
    @staticmethod
    def _get_ensemble_defaults(ensemble: str) -> dict[str, Any]:
        """Get default params for the ensemble."""
        defaults = {
            "nve": {"algorithm": "verlet"},
            "nvt": {"algorithm": "nose-hoover"},
            "npt": {"algorithm": "nose-hoover"},
        }

        try:
            return defaults[ensemble.lower()]  # type: ignore[return-value]
        except KeyError as err:
            supported = tuple(defaults)
            raise ValueError(f"Expect {ensemble=} to be one of {supported}") from err
