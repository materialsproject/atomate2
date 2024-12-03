from atomate2.lammps.sets.core import BaseLammpsSet, LammpsNVTSet, LammpsNPTSet, LammpsMinimizeSet

def test_LammpsNVTSet():
    nvt = BaseLammpsSet()
    nvt.update_settings(timestep=0.005, ensemble='nvt', thermostat='langevin')
    assert nvt.ensemble.value == 'nvt'
    assert nvt.settings['thermostat'] == 'langevin'
    assert nvt.settings['timestep'] == 0.005
    
def test_minimize_set():
    mini = LammpsMinimizeSet()
    assert mini.ensemble.value == 'npt'