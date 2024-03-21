from typing import Optional
from dataclasses import dataclass

import numpy as np

from atomate2.classical_md.openmm.jobs.base import BaseOpenMMMaker

from openmm import LangevinMiddleIntegrator
from openmm.openmm import MonteCarloBarostat
from openmm.unit import kelvin, atmosphere, picoseconds, kilojoules_per_mole, nanometer

from atomate2.classical_md.utils import create_array_summing_to


@dataclass
class EnergyMinimizationMaker(BaseOpenMMMaker):
    name: str = "energy minimization"
    steps: int = 0
    tolerance: float = 10
    max_iterations: int = 0

    def run_openmm(self, sim):

        assert self.steps == 0, "Energy minimization should have 0 steps."

        # Minimize the energy
        sim.minimizeEnergy(
            tolerance=self.tolerance * kilojoules_per_mole / nanometer,
            maxIterations=self.max_iterations,
        )


@dataclass
class NPTMaker(BaseOpenMMMaker):
    name: str = "npt simulation"
    steps: int = 1000000
    pressure: float = 1
    step_size: Optional[float] = None
    temperature: Optional[float] = None
    friction_coefficient: Optional[float] = None
    pressure_update_frequency: int = 10

    def run_openmm(self, sim):
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
    name: str = "nvt simulation"
    steps: int = 1000000
    step_size: Optional[float] = None
    temperature: Optional[float] = None
    friction_coefficient: Optional[float] = None

    def run_openmm(self, sim):

        # Run the simulation
        sim.step(self.steps)


@dataclass
class TempChangeMaker(BaseOpenMMMaker):
    name: str = "temperature change"
    steps: int = 1000000
    temp_steps: int = 100
    step_size: Optional[float] = None
    temperature: Optional[float] = None
    friction_coefficient: Optional[float] = None
    starting_temperature: Optional[float] = None

    def run_openmm(self, sim):
        integrator = sim.context.getIntegrator()

        temps_arr = np.linspace(
            self.starting_temperature, self.temperature, self.temp_steps
        )
        steps_arr = create_array_summing_to(self.steps, self.temp_steps)
        for temp, n_steps in zip(temps_arr, steps_arr):
            integrator.setTemperature(temp * kelvin)
            sim.step(n_steps)

    def _create_integrator(self, prev_task) -> LangevinMiddleIntegrator:
        # we do this dance so _resolve_attr takes its value from the previous task
        temp_holder, self.temperature = self.temperature, None
        self.starting_temperature = self._resolve_attr("temperature", prev_task)
        self.temperature = temp_holder or self._resolve_attr("temperature", prev_task)
        return LangevinMiddleIntegrator(
            self.starting_temperature * kelvin,
            self._resolve_attr("friction_coefficient", prev_task) / picoseconds,
            self._resolve_attr("step_size", prev_task) * picoseconds,
        )
