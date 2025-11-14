from atomate2.lammps.sets.core import LammpsMinimizeSet, LammpsNVTSet, LammpsNPTSet


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


def test_npt_set():
    npt = LammpsNPTSet(
        settings={
            "barostat": "nose-hoover",
            "timestep": 0.005,
            "start_pressure": 1.0,
            "end_pressure": 1.0,
            "start_temp": 300,
            "end_temp": 300,
            "psymm": "iso",
        }
    )
    if isinstance(npt.settings, dict):
        assert npt.settings["ensemble"] == "npt"
        assert npt.settings["barostat"] == "nose-hoover"
        assert npt.settings["timestep"] == 0.005
        assert npt.settings["start_pressure"] == 1.0
        assert npt.settings["end_pressure"] == 1.0
        assert npt.settings["start_temp"] == 300
        assert npt.settings["end_temp"] == 300
        assert npt.settings["psymm"] == "iso"
    else:
        assert npt.settings.ensemble == "npt"
        assert npt.settings.barostat == "nose-hoover"
        assert npt.settings.timestep == 0.005
        assert npt.settings.start_pressure == 1.0
        assert npt.settings.end_pressure == 1.0
        assert npt.settings.start_temp == 300
        assert npt.settings.end_temp == 300
        assert npt.settings.psymm == "iso"
