from atomate2.openmm.jobs.base_openmm_maker import BaseOpenMMMaker
from atomate2.openmm.schemas.task_details import TaskDetails
from dataclasses import dataclass, asdict
from openmm.unit import kelvin
import numpy as np


@dataclass
class TempChangeMaker(BaseOpenMMMaker):
    steps: int = 1000000
    name: str = "temperature change"
    final_temp: float = 298
    temp_steps: int = 100

    def _run_openmm(self, sim):
        # Add barostat to system
        integrator = sim.context.getIntegrator()
        start_temp = integrator.getTemperature()

        # Heating temperature
        delta_t = abs(self.final_temp * kelvin - start_temp)
        if delta_t < 1e-6 * kelvin:
            raise ValueError(f"Final temperature {self.final_temp} is too close to "
                             f"starting temperature {start_temp}, make sure the "
                             f"TempChangeMaker has a temperature differential.")
        temp_step_size = delta_t / self.temp_steps
        for temp in np.arange(
                start_temp + temp_step_size,
                self.final_temp * kelvin + temp_step_size,
                temp_step_size,
        ):
            integrator.setTemperature(temp * kelvin)
            sim.step(self.steps // self.temp_steps)

        # TODO: we could also write out task_details like this?
        task_details = asdict(self)

        task_details = TaskDetails(
            task_name=self.name,
            task_kwargs={
                **asdict(self)
            },
            platform_kwargs=self.platform_kwargs,
            total_steps=self.steps,
        )
        return TaskDetails.from_maker(self)
