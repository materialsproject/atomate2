import os
from atomate2.lammps.sets.base import BaseLammpsSet, template_dir, _BASE_LAMMPS_SETTINGS
from typing import Literal
from atomate2.ase.md import MDEnsemble

class LammpsNVTSet(BaseLammpsSet):
    """
    Lammps input set for NVT MD simulations. 
    """
    ensemble : MDEnsemble = MDEnsemble.nvt
    friction : float = None
    thermostat : str = "langevin"
    settings : dict = None
    
    def __init__(self, 
                 thermostat : Literal['langevin', 'nose-hoover'] = 'langevin',
                 friction : float = None,
                 **kwargs):
        
        self.thermostat = thermostat
        if friction is None:
            friction = 100*kwargs.get('timestep', _BASE_LAMMPS_SETTINGS['timestep'])
        if friction < kwargs.get('timestep', 0.001):
            raise ValueError("Friction should be more than the timestep!")
        
        self.settings = {'thermostat': self.thermostat, 
                         'tfriction': friction,
                         'ensemble': self.ensemble.value
                         }
        super().__init__(ensemble=self.ensemble, thermostat=self.thermostat, settings=self.settings, **kwargs)

class LammpsNPTSet(BaseLammpsSet):
    """
    Lammps input set for NPT MD simulations.
    """
    ensemble : MDEnsemble = MDEnsemble.npt
    friction : float = None
    barostat : str = "nose-hoover"
    settings : dict = None
    
    def __init__(self, 
                 barostat : Literal['berendsen', 'nose-hoover'] = 'nose-hoover',
                 friction : float = None,
                 **kwargs):
        
        self.barostat = barostat
 
        if friction is None:
            friction = 100*kwargs.get('timestep', _BASE_LAMMPS_SETTINGS['timestep'])
        if friction < kwargs.get('timestep', 0.001):
            raise ValueError("Friction should be more than the timestep!")
        
        self.settings = {'barostat': barostat, 
                         'pfriction': friction,
                         'ensemble': self.ensemble.value
                         }
        super().__init__(ensemble=self.ensemble, barostat=self.barostat, settings=self.settings, **kwargs)
        
        
class LammpsMinimizeSet(BaseLammpsSet):
    """
    Input set for minimization simulations.
    """
    settings : dict = {'ensemble': 'minimize'}
    
    def __init__(self, **kwargs):
        template = os.path.join(template_dir, "minimization.template")
        super().__init__(ensemble='npt', template=template, settings=self.settings, **kwargs)
    