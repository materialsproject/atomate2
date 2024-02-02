from dataclasses import dataclass
from ase.calculators import calculator


# This needs the most help with... I'm not sure what to do with it.
@dataclass
class ASEMDInputs:
    calculator: calculator
    timestep: float = 1.0
    steps: int = 50
    temperature: float = 300.0
    friction: float = 0.01
    trajectory_fn: str = "out.traj"
    logfile_fn: str = "out.log"
    loginterval: int = 10
    append_trajectory: bool = False
    save_files: bool = True
