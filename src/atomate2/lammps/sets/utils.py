from monty.serialization import loadfn
import os

settings_dir = os.path.dirname(os.path.abspath(__file__))
_BASE_LAMMPS_SETTINGS = loadfn(os.path.join(settings_dir,'BASE_LAMMPS_SETTINGS.json'))
all_settings_keys = loadfn(os.path.join(settings_dir, 'lammps_settings_keys.json'))

def update_settings(settings : dict = None, **kwargs):
    """
    Update the settings for the LAMMPS input file.
    """
    
    base_settings = _BASE_LAMMPS_SETTINGS.copy()
        
    if settings is None:
        settings = base_settings
    
    for k in base_settings.keys():
        if k not in kwargs.keys():
            settings.update({k: base_settings.get(k)})
    
    return base_settings


def process_ensemble_conditions(settings : dict):
        """
        Process the ensemble conditions for the LAMMPS input file.
        """
        ensemble = settings.get('ensemble', 'nve')
        for k in ["nve", "nvt", "npt"]:
            settings.update({f'{k}_flag': '#'})
            
        if ensemble == 'nvt':
            if settings.get('thermostat') == 'langevin':
                settings.update({'nve_flag': 'fix', 'thermseed': settings.get('seed', 0)})
                
            else: 
                settings.update({'thermostat': 'nvt temp'})
        
            
        if ensemble == 'npt':
            if settings.get('barostat') == 'berendsen':
                settings.update({'nve_flag': 'fix', 'barostat': 'press/berendsen'})
            else:
                settings.update({'barostat': 'npt temp'})
        
        if ensemble == 'nph':
            settings.update({'npt_flag': 'fix', 'barostat': 'nph'})
        
        settings.update({f'{ensemble}_flag': 'fix'}) if ensemble in ["nve", "nvt", "npt"] else None
        return settings