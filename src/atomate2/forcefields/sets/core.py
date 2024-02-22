"""Module defining core ASE MD input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pymatgen.io.core import InputGenerator, InputSet

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
    temperature : float or list of floats or None
        Temperature schedule for the simulation in Kelvin. Default is None. If None,
        the temperature is constant at the value of 300 K. If a list, the temperature
        schedule is interpolated in equidistant steps for the simulation.
    pressure
        Pressure schedule for the simulation in kilobar. Default is None. If None,
        the pressure is constant at the value of 0. Positive values are for compression.
        Negative values are for expansion. If a list, the pressure schedule is
        interpolated in equidistant steps for the simulation.
    **kwargs
        Other parameters passed to the InputGenerator.
    """

    ensemble: str = "nvt"
    nsteps: int = 1000
    timestep: float = 2
    temperature: float | list[float] | None = None
    pressure: float | list[float] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    # TODO: revise and pay attention to the order of default and overriden values
    def get_input_set(self) -> InputSet:
        """
        Get the input set.

        Returns
        -------
        InputSet
            The input set.
        """
        raise NotImplementedError
        # inputs = self._get_ensemble_defaults(self.ensemble)
        # inputs.update(self.kwargs)
        # return InputSet(inputs=inputs)

    # TODO: add more detials
    @staticmethod
    def _get_ensemble_defaults(ensemble: str) -> dict[str, Any]:
        """Get default params for the ensemble."""
        defaults = {
            "nve": {"algorithm": "verlet"},
            "nvt": {"algorithm": "nose-hoover"},
            "npt": {"algorithm": "nose-hoover"},
        }

        try:
            return defaults[ensemble.lower()]
        except KeyError as err:
            supported = tuple(defaults)
            raise ValueError(f"Expect {ensemble=} to be one of {supported}") from err
