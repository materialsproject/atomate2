"""Core LAMMPS input set generators."""

from typing import Literal

from pymatgen.io.lammps.generators import _BASE_LAMMPS_SETTINGS, BaseLammpsSetGenerator

from atomate2.ase.md import MDEnsemble


class LammpsNVESet(BaseLammpsSetGenerator):
    """Lammps input set for NVE MD simulations."""

    ensemble: MDEnsemble = MDEnsemble.nve
    settings: dict | None = None

    def __init__(self, settings: dict | None = None, **kwargs) -> None:
        self.settings = settings or {}
        self.settings.update(settings)
        self.settings.update(
            {k: v for k, v in kwargs.items() if k in _BASE_LAMMPS_SETTINGS}
        )
        kwargs = {k: v for k, v in kwargs.items() if k not in _BASE_LAMMPS_SETTINGS}
        self.settings.update(
            {
                "ensemble": self.ensemble.value,
                "thermostat": None,
                "barostat": None,
                "friction": None,
            }
        )

        super().__init__(
            calc_type=f"lammps_{self.ensemble.value}", settings=self.settings, **kwargs
        )


class LammpsNVTSet(BaseLammpsSetGenerator):
    """Lammps input set for NVT MD simulations."""

    ensemble: MDEnsemble = MDEnsemble.nvt
    friction: float = None
    thermostat: Literal["langevin", "nose-hoover"] = "langevin"
    start_temp: float = 300.0
    end_temp: float = 300.0
    settings: dict | None = None

    def __init__(
        self,
        thermostat: Literal["langevin", "nose-hoover"] = "langevin",
        start_temp: float = 300.0,
        end_temp: float = 300.0,
        friction: float = 0.1,
        settings: dict | None = None,
        **kwargs,
    ) -> None:
        self.friction = friction
        self.thermostat = thermostat
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.settings = settings or {}
        self.settings.update(settings)

        self.settings.update(
            {k: v for k, v in kwargs.items() if k in _BASE_LAMMPS_SETTINGS}
        )
        kwargs = {k: v for k, v in kwargs.items() if k not in _BASE_LAMMPS_SETTINGS}

        self.settings.update(
            {
                "ensemble": self.ensemble.value,
                "thermostat": self.thermostat,
                "start_temp": start_temp,
                "end_temp": end_temp,
                "friction": friction,
            }
        )

        super().__init__(
            calc_type=f"lammps_{self.ensemble.value}", settings=self.settings, **kwargs
        )


class LammpsNPTSet(BaseLammpsSetGenerator):
    """Lammps input set for NPT MD simulations."""

    ensemble: MDEnsemble = MDEnsemble.npt
    friction: float = None
    barostat: Literal["berendsen", "nose-hoover", "nph"] = "nose-hoover"
    start_pressure: float = 1.0
    end_pressure: float = 1.0
    start_temp: float = 300
    end_temp: float = 300
    settings: dict | None = None

    def __init__(
        self,
        barostat: Literal["berendsen", "nose-hoover", "nph"] = "nose-hoover",
        friction: float = 0.1,
        start_pressure: float = 1.0,
        end_pressure: float = 1.0,
        start_temp: float = 300,
        end_temp: float = 300,
        settings: dict | None = None,
        **kwargs,
    ) -> None:
        self.barostat = barostat
        self.friction = friction
        self.start_pressure = start_pressure
        self.end_pressure = end_pressure
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.settings = settings or {}
        self.settings.update(
            {
                "ensemble": self.ensemble.value,
                "barostat": self.barostat,
                "start_pressure": start_pressure,
                "end_pressure": end_pressure,
                "start_temp": start_temp,
                "end_temp": end_temp,
                "friction": friction,
            }
        )
        self.settings.update(settings)
        self.settings.update(
            {k: v for k, v in kwargs.items() if k in _BASE_LAMMPS_SETTINGS}
        )
        kwargs = {k: v for k, v in kwargs.items() if k not in _BASE_LAMMPS_SETTINGS}

        super().__init__(
            calc_type=f"lammps_{self.ensemble.value}", settings=self.settings, **kwargs
        )


class LammpsMinimizeSet(BaseLammpsSetGenerator):
    """Input set for minimization simulations."""

    settings: dict | None = None
    max_steps: int = 10000
    pressure: float = 0
    tol: float = 1.0e-6

    def __init__(
        self,
        max_steps: int = 10000,
        pressure: float = 0,
        tol: float = 1.0e-6,
        settings: dict | None = None,
        **kwargs,
    ) -> None:
        self.max_steps = max_steps
        self.tol = tol
        self.pressure = pressure
        self.settings = settings or {}
        self.settings.update(
            {
                "ensemble": "minimize",
                "nsteps": self.max_steps,
                "start_pressure": self.pressure,
                "end_pressure": self.pressure,
                "tol": self.tol,
            }
        )
        self.settings.update(settings)
        self.settings.update(
            {k: v for k, v in kwargs.items() if k in _BASE_LAMMPS_SETTINGS}
        )
        kwargs = {k: v for k, v in kwargs.items() if k not in _BASE_LAMMPS_SETTINGS}

        super().__init__(
            calc_type="lammps_minimization", settings=self.settings, **kwargs
        )
