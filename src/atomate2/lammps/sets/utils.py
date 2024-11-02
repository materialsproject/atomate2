from monty.serialization import loadfn
import os

settings_dir = os.path.dirname(os.path.abspath(__file__))
_BASE_LAMMPS_SETTINGS = loadfn(os.path.join(settings_dir,'BASE_LAMMPS_SETTINGS.json'))
all_settings_keys = loadfn(os.path.join(settings_dir, 'lammps_settings_keys.json'))

def update_settings(settings : dict = None, **kwargs):
    """
    Update the settings for the LAMMPS input file.
    """
    if settings is None:
        settings = _BASE_LAMMPS_SETTINGS.copy()
        
    for k, v in kwargs.items():
        if k in all_settings_keys:
            settings.update({k: v})
        else:
            Warning("Invalid key {k} for LAMMPS settings, skipping...")
    
    return settings


def process_ensemble_conditions(settings : dict):
        """
        Process the ensemble conditions for the LAMMPS input file.
        """
        ensemble = settings.get('ensemble', 'nve')
        for k in ["nve", "nvt", "npt"]:
            settings.update({f'{k}_flag': '#'})
            
        settings.update({f'{ensemble}_flag': 'fix'}) if ensemble in ["nve", "nvt", "npt"] else None
        return settings