from atomate2.lammps.sets.core import LammpsMinimizeSet, LammpsNVTSet


def test_nvt_set():
    nvt = LammpsNVTSet(settings={"thermostat": "langevin", "timestep": 0.005})
    if isinstance(nvt.settings, dict):
        assert nvt.settings["ensemble"] == "nvt"
        assert nvt.settings["thermostat"] == "langevin"
        assert nvt.settings["timestep"] == 0.005
    else:
        assert nvt.settings.ensemble == "nvt"
        assert nvt.settings.thermostat == "langevin"
        assert nvt.settings.timestep == 0.005


def test_minimize_set():
    mini = LammpsMinimizeSet()
    if isinstance(mini.settings, dict):
        assert mini.settings["ensemble"] == "minimize"
    else:
        assert mini.settings.ensemble == "minimize"
