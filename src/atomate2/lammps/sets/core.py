import os
from .base import BaseLammpsSet, template_dir
from atomate2.lammps.sets.utils import update_settings, _BASE_LAMMPS_SETTINGS
from typing import Literal, Union
from atomate2.ase.md import MDEnsemble

class LammpsNVTSet(BaseLammpsSet):
    """
    Lammps input set for NVT MD simulations. 
    """
    ensemble : MDEnsemble = MDEnsemble.nvt
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
            raise ValueError("Friction should be more than the timestep")
        
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
            raise ValueError("Friction should be more than the timestep")
        
        self.settings = {'barostat': barostat, 
                         'pfriction': friction,
                         'ensemble': self.ensemble.value
                         }
        
        self.settings = update_settings(settings=self.settings,**kwargs)
        super().__init__(ensemble=self.ensemble, barostat=self.barostat, settings=self.settings, **kwargs)
        
        
class LammpsMinimizeSet(BaseLammpsSet):
    """
    Input set for minimization simulations.
    """
    setting : dict = {'ensemble': 'minimize'}
    
    def __init__(self, **kwargs):
        template = os.path.join(template_dir, "minimize.template")
        super().__init__(ensemble='minimize', template=template, settings=self.settings, **kwargs)
    