"""Core OpenMM jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from openmm import Integrator, LangevinMiddleIntegrator, MonteCarloBarostat
from openmm.unit import atmosphere, kelvin, kilojoules_per_mole, nanometer, picoseconds

from atomate2.classical_md.openmm.jobs.base import BaseOpenMMMaker
from atomate2.classical_md.utils import create_list_summing_to

if TYPE_CHECKING:
    from openmm.app import Simulation

    from atomate2.classical_md.openmm.schemas.tasks import OpenMMTaskDocument


@dataclass
class EnergyMinimizationMaker(BaseOpenMMMaker):
    """
    A maker class for performing energy minimization using OpenMM.

    This class inherits from BaseOpenMMMaker, only new attributes are documented.

    Attributes
    ----------
        name (str): The name of the energy minimization job.
            Default is "energy minimization".
        steps (int): The number of minimization steps. Must be equal to 0.
        tolerance (float): The energy tolerance for minimization. Default is 10 kj/nm.
        max_iterations (int): The maximum number of minimization iterations.
        Default is 0, which means no maximum.
    """

    name: str = "energy minimization"
    steps: int = 0
    tolerance: float = 10
    max_iterations: int = 0

    def run_openmm(self, sim: Simulation) -> None:
        """
        Run the energy minimization with OpenMM.

        This method performs energy minimization on the molecular system using
        the OpenMM simulation package. It minimizes the energy of the system
        based on the specified tolerance and maximum number of iterations.

        Args:
            sim (Simulation): The OpenMM simulation object.
        """
        if self.steps != 0:
            raise ValueError("Energy minimization should have 0 steps.")

        # Minimize the energy
        sim.minimizeEnergy(
            tolerance=self.tolerance * kilojoules_per_mole / nanometer,
            maxIterations=self.max_iterations,
        )


@dataclass
class NPTMaker(BaseOpenMMMaker):
    """
    A maker class for performing NPT (isothermal-isobaric) simulations using OpenMM.

    This class inherits from BaseOpenMMMaker, only new attributes are documented.

    Attributes
    ----------
        name (str): The name of the NPT simulation job. Default is "npt simulation".
        steps (int): The number of simulation steps. Default is 1000000.
        pressure (float): The pressure of the simulation in atmospheres.
            Default is 1 atm.
        pressure_update_frequency (int): The number of steps between pressure
            update attempts.
    """

    name: str = "npt simulation"
    steps: int = 1000000
    pressure: float = 1
    pressure_update_frequency: int = 10

    def run_openmm(self, sim: Simulation) -> None:
        """
        Evolve the simulation for self.steps in the NPT ensemble.

        This adds a Monte Carlo barostat to the system to put it into NPT, runs the
        simulation for the specified number of steps, and then removes the barostat.

        Args:
        sim (Simulation): The OpenMM simulation object.
        """
        # Add barostat to system
        context = sim.context
        system = context.getSystem()

        barostat_force_index = system.addForce(
            MonteCarloBarostat(
                self.pressure * atmosphere,
                sim.context.getIntegrator().getTemperature(),
                self.pressure_update_frequency,
            )
        )

        # Re-init the context after adding thermostat to System
        context.reinitialize(preserveState=True)

        # Run the simulation
        sim.step(self.steps)

        # Remove thermostat and update context
        system.removeForce(barostat_force_index)
        context.reinitialize(preserveState=True)


@dataclass
class NVTMaker(BaseOpenMMMaker):
    """
    A maker class for performing NVT (canonical ensemble) simulations using OpenMM.

    This class inherits from BaseOpenMMMaker, only new attributes are documented.

    Attributes
    ----------
        name (str): The name of the NVT simulation job. Default is "nvt simulation".
        steps (int): The number of simulation steps. Default is 1000000.
    """

    name: str = "nvt simulation"
    steps: int = 1000000

    def run_openmm(self, sim: Simulation) -> None:
        """
        Evolve the simulation with OpenMM for self.steps.

        Args:
            sim (Simulation): The OpenMM simulation object.
        """
        # Run the simulation
        sim.step(self.steps)


@dataclass
class TempChangeMaker(BaseOpenMMMaker):
    """
    A maker class for performing simulations with temperature changes using OpenMM.

    This class inherits from BaseOpenMMMaker and provides
    functionality for running simulations with temperature
    changes using the OpenMM simulation package.

    Attributes
    ----------
    name (str): The name of the temperature change job. Default is "temperature change".
    steps (int): The total number of simulation steps. Default is 1000000.
    temp_steps (int): The number of steps over which the temperature is raised, by
        default will be set to steps / 10000.
    starting_temperature (Optional[float]): The starting temperature of the simulation.
        If not provided it will inherit from the previous task.
    """

    name: str = "temperature change"
    steps: int = 1000000
    temp_steps: int | None = None
    starting_temperature: float | None = None

    def run_openmm(self, sim: Simulation) -> None:
        """
        Evolve the simulation while gradually changing the temperature.

        self.temperature is the final temperature. self.temp_steps
        determines how many gradiations there are between the starting and
        final temperature. At each gradiation, the system is evolved for a
        number of steps such that the total number of steps is self.steps.

        Args:
            sim (Simulation): The OpenMM simulation object.
        """
        integrator = sim.context.getIntegrator()

        temps_arr = np.linspace(
            self.starting_temperature, self.temperature, self.temp_steps
        )
        steps_list = create_list_summing_to(self.steps, self.temp_steps)
        for temp, n_steps in zip(temps_arr, steps_list):
            integrator.setTemperature(temp * kelvin)
            sim.step(n_steps)

    def _create_integrator(
        self, prev_task: OpenMMTaskDocument | None = None
    ) -> Integrator:
        # we resolve this here because prev_task is available
        temp_steps_default = (self.steps // 10000) or 1
        self.temp_steps = self._resolve_attr(
            "temp_steps", prev_task, add_defaults={"temp_steps": temp_steps_default}
        )

        # we do this dance so _resolve_attr takes its value from the previous task
        temp_holder, self.temperature = self.temperature, None
        self.starting_temperature = self._resolve_attr("temperature", prev_task)
        self.temperature = temp_holder or self._resolve_attr("temperature", prev_task)
        return LangevinMiddleIntegrator(
            self.starting_temperature * kelvin,
            self._resolve_attr("friction_coefficient", prev_task) / picoseconds,
            self._resolve_attr("step_size", prev_task) * picoseconds,
        )
