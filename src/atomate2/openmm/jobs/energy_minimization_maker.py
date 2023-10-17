from atomate2.openmm.jobs.base_openmm_maker import BaseOpenMMMaker
from atomate2.openmm.schemas.task_details import TaskDetails
from dataclasses import dataclass


@dataclass
class EnergyMinimizationMaker(BaseOpenMMMaker):
    name: str = "energy minimization"
    # TODO: add default kwargs for Simulation.minimizeEnergy?
    # tolerance
    # maxIterations : int

    def _run_openmm(self, sim):

        # Minimize the energy
        sim.minimizeEnergy()
        return TaskDetails.from_maker(self)
