from atomate2.lammps.sets.base import BaseLammpsSetGenerator
from typing import Literal
from atomate2.ase.md import MDEnsemble
import numpy as np


class LammpsNVESet(BaseLammpsSetGenerator):
    """
    Lammps input set for NVE MD simulations. 
    """
    ensemble : MDEnsemble = MDEnsemble.nve
    settings : dict = {}
    
    def __init__(self, **kwargs):
        self.settings.update({'ensemble': self.ensemble.value})
        
        super().__init__(calc_type=f'lammps_{self.ensemble.value}', settings=self.settings, **kwargs)


class LammpsNVTSet(BaseLammpsSetGenerator):
    """
    Lammps input set for NVT MD simulations. 
    """
    ensemble : MDEnsemble = MDEnsemble.nvt
    friction : float = None
    temperature : float | list | np.ndarray = 300
    thermostat : Literal['langevin', 'nose-hoover'] = "langevin"
    settings : dict = {}
    
    def __init__(self, 
                 thermostat : Literal['langevin', 'nose-hoover'] = 'langevin',
                 temperature : float | list | np.ndarray = 300,
                 friction : float = None,
                 **kwargs):
        self.friction = friction
        self.thermostat = thermostat                     
        self.settings.update({'ensemble': self.ensemble.value,
                              'thermostat': self.thermostat,
                              'temperature': temperature})
        
        super().__init__(calc_type=f'lammps_{self.ensemble.value}', settings=self.settings, **kwargs)

class LammpsNPTSet(BaseLammpsSetGenerator):
    """
    Lammps input set for NPT MD simulations.
    """
    ensemble : MDEnsemble = MDEnsemble.npt
    friction : float = None
    pressure : float | list | np.ndarray = 1.0
    temperature : float | list = 300
    barostat : Literal['berendsen', 'nose-hoover', 'nph'] = "nose-hoover"
    settings : dict = {}
    
    def __init__(self, 
                 barostat : Literal['berendsen', 'nose-hoover', 'nph'] = 'nose-hoover',
                 pressure : float | list | np.ndarray = 1.0,
                 temperature : float | list = 300,
                 friction : float = None,
                 **kwargs):
        
        self.barostat = barostat
        self.pressure = pressure
        self.temperature = temperature
        self.friction = friction
        self.settings = {'ensemble': self.ensemble.value,
                            'barostat': self.barostat,
                            'pressure': self.pressure,
                            'temperature': self.temperature}
        
        super().__init__(calc_type=f'lammps_{self.ensemble.value}', settings=self.settings, **kwargs)
        
        
class LammpsMinimizeSet(BaseLammpsSetGenerator):
    """
    Input set for minimization simulations.
    """
    settings : dict = None
    pressure : float | list | np.ndarray = 0.0
    max_steps : int = 10000
    tol : float = 1.0e-6
    
    def __init__(self,
                 pressure : float | list | np.ndarray = 0.0,
                 max_steps : int = 10000,
                 tol : float = 1.0e-6,
                 **kwargs):
        
        self.pressure = pressure
        self.max_steps = max_steps
        self.tol = tol
        
        self.settings = {'ensemble': 'minimize',
                         'pressure': self.pressure,
                         'max_steps': self.max_steps,
                         'tol': self.tol}
            
        super().__init__(calc_type='lammps_minimization', settings=self.settings, **kwargs)
    