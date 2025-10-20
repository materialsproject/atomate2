"""Core LAMMPS input set generators."""

from dataclasses import dataclass, field

from pymatgen.io.lammps.generators import (
    _BASE_LAMMPS_SETTINGS,
    BaseLammpsSetGenerator,
    LammpsSettings,
)

from atomate2.ase.md import MDEnsemble


@dataclass
class LammpsNVESet(BaseLammpsSetGenerator):
    """Lammps input set for NVE MD simulations.

    All configuration parameters are passed through the `settings` dict.
    The ensemble-specific defaults will be applied automatically.

    Args:
        settings: Dictionary containing LAMMPS settings. Common options include:
            - timestep (float): Simulation timestep. Default: 0.001 ps
            - nsteps (int): Number of simulation steps. Default: 1000
            - log_interval (int): Thermodynamic logging interval. Default: 100
            - traj_interval (int): Trajectory output interval. Default: 100
            - Any other LAMMPS settings from _BASE_LAMMPS_SETTINGS
        force_field: Force field file or dictionary (inherited from base class)
        inputfile: Custom input file (inherited from base class)
        Other base class parameters as needed

    Example:
        >>> nve = LammpsNVESet(
        ...     settings={"timestep": 0.001, "nsteps": 10000, "log_interval": 500}
        ... )
    """

    ensemble: MDEnsemble = field(default=MDEnsemble.nve)
    settings: LammpsSettings | dict | None = field(default=None)

    def __post_init__(self) -> None:
        """Initialize NVE-specific settings and defaults."""
        self.calc_type = f"lammps_{self.ensemble.value}"
        # Initialize settings if None
        if self.settings is None:
            settings_dict = {}
        elif isinstance(self.settings, LammpsSettings):
            settings_dict = self.settings.as_dict()
        else:
            settings_dict = self.settings.copy()
        # Add ensemble-specific defaults directly to self.settings
        settings_dict.update(
            {
                "ensemble": self.ensemble.value,
                "thermostat": None,
                "barostat": None,
                "friction": None,
            }
        )
        self.settings = settings_dict
        super().__post_init__()


@dataclass
class LammpsNVTSet(BaseLammpsSetGenerator):
    """Lammps input set for NVT MD simulations.

    All configuration parameters are passed through the `settings` dict.
    The ensemble-specific defaults will be applied automatically.

    Args:
        settings: Dictionary containing LAMMPS settings. NVT-specific options include:
            - thermostat (str): Thermostat type. Options: "langevin", "nose-hoover".
              Default: "langevin"
            - start_temp (float): Initial temperature in K. Default: 300.0
            - end_temp (float): Final temperature in K. Default: 300.0
            - friction (float): Thermostat friction coefficient. Default: 0.1 ps^-1
            - timestep (float): Simulation timestep. Default: 0.001 ps
            - nsteps (int): Number of simulation steps. Default: 1000
            - log_interval (int): Thermodynamic logging interval. Default: 100
            - traj_interval (int): Trajectory output interval. Default: 100
            - Any other LAMMPS settings from _BASE_LAMMPS_SETTINGS
        force_field: Force field file or dictionary (inherited from base class)
        inputfile: Custom input file (inherited from base class)
        Other base class parameters as needed

    Example:
        >>> nvt = LammpsNVTSet(
        ...     settings={
        ...         "thermostat": "langevin",
        ...         "start_temp": 300,
        ...         "end_temp": 1000,
        ...         "friction": 0.1,
        ...         "timestep": 0.001,
        ...         "nsteps": 100000,
        ...     }
        ... )
    """

    ensemble: MDEnsemble = field(default=MDEnsemble.nvt)
    settings: LammpsSettings | dict | None = field(default=None)

    def __post_init__(self) -> None:
        """Initialize NVT-specific settings and defaults."""
        self.calc_type = f"lammps_{self.ensemble.value}"
        # Initialize settings if None
        if self.settings is None:
            settings_dict = {}
        elif isinstance(self.settings, LammpsSettings):
            settings_dict = self.settings.as_dict()
        else:
            settings_dict = self.settings.copy()

        # Add ensemble-specific defaults, using values from settings if provided
        settings_dict.update(
            {
                "ensemble": self.ensemble.value,
                "thermostat": settings_dict.get("thermostat", "langevin"),
                "start_temp": settings_dict.get("start_temp", 300.0),
                "end_temp": settings_dict.get("end_temp", 300.0),
                "friction": settings_dict.get(
                    "friction", _BASE_LAMMPS_SETTINGS["periodic"]["friction"]
                ),
            }
        )

        # Set the updated dict back to self.settings
        self.settings = settings_dict
        super().__post_init__()


@dataclass
class LammpsNPTSet(BaseLammpsSetGenerator):
    """Lammps input set for NPT MD simulations.

    All configuration parameters are passed through the `settings` dict.
    The ensemble-specific defaults will be applied automatically.

    Args:
        settings: Dictionary containing LAMMPS settings. NPT-specific options include:
            - barostat (str): Barostat type. Options: "berendsen", "nose-hoover", "nph".
              Default: "nose-hoover"
            - start_pressure (float): Initial pressure in atm. Default: 1.0
            - end_pressure (float): Final pressure in atm. Default: 1.0
            - start_temp (float): Initial temperature in K. Default: 300
            - end_temp (float): Final temperature in K. Default: 300
            - friction (float): Thermostat/barostat friction coefficient.
              Default: 0.1 ps^-1
            - timestep (float): Simulation timestep. Default: 0.001 ps
            - nsteps (int): Number of simulation steps. Default: 1000
            - log_interval (int): Thermodynamic logging interval. Default: 100
            - traj_interval (int): Trajectory output interval. Default: 100
            - Any other LAMMPS settings from _BASE_LAMMPS_SETTINGS
        force_field: Force field file or dictionary (inherited from base class)
        inputfile: Custom input file (inherited from base class)
        Other base class parameters as needed

    Example:
        >>> npt = LammpsNPTSet(
        ...     settings={
        ...         "barostat": "nose-hoover",
        ...         "start_pressure": 1.0,
        ...         "end_pressure": 10.0,
        ...         "start_temp": 300,
        ...         "end_temp": 1000,
        ...         "friction": 0.1,
        ...         "timestep": 0.001,
        ...         "nsteps": 100000,
        ...     }
        ... )
    """

    ensemble: MDEnsemble = field(default=MDEnsemble.npt)
    settings: LammpsSettings | dict | None = field(default=None)

    def __post_init__(self) -> None:
        """Initialize NPT-specific settings and defaults."""
        self.calc_type = f"lammps_{self.ensemble.value}"
        # Initialize settings if None
        if self.settings is None:
            settings_dict = {}
        elif isinstance(self.settings, LammpsSettings):
            settings_dict = self.settings.as_dict()
        else:
            settings_dict = self.settings.copy()

        # Add ensemble-specific defaults, using values from settings if provided
        settings_dict.update(
            {
                "ensemble": self.ensemble.value,
                "barostat": settings_dict.get("barostat", "nose-hoover"),
                "start_pressure": settings_dict.get("start_pressure", 1.0),
                "end_pressure": settings_dict.get("end_pressure", 1.0),
                "start_temp": settings_dict.get("start_temp", 300),
                "end_temp": settings_dict.get("end_temp", 300),
                "friction": settings_dict.get(
                    "friction", _BASE_LAMMPS_SETTINGS["periodic"]["friction"]
                ),
            }
        )

        # Set the updated dict back to self.settings
        self.settings = settings_dict
        super().__post_init__()


@dataclass
class LammpsMinimizeSet(BaseLammpsSetGenerator):
    """Input set for minimization simulations.

    All configuration parameters are passed through the `settings` dict.
    The ensemble-specific defaults will be applied automatically.

    Args:
        settings: Dictionary containing LAMMPS settings.
        Minimization-specific options include:
            - nsteps (int): Maximum number of minimization steps. Default: 10000
            - start_pressure (float): Initial pressure in atm. Default: 0
            - end_pressure (float): Final pressure in atm. Default: 0
            - tol (float): Convergence tolerance. Default: 1.0e-6
            - min_style (str): Minimization algorithm.
            Options: "cg", "sd", "fire", etc. Default: "cg"
            - timestep (float): Simulation timestep. Default: 0.001 ps
            - log_interval (int): Thermodynamic logging interval. Default: 100
            - traj_interval (int): Trajectory output interval. Default: 100
            - Any other LAMMPS settings from _BASE_LAMMPS_SETTINGS
        force_field: Force field file or dictionary (inherited from base class)
        inputfile: Custom input file (inherited from base class)
        Other base class parameters as needed

    Example:
        >>> mini = LammpsMinimizeSet(
        ...     settings={
        ...         "nsteps": 50000,
        ...         "tol": 1.0e-8,
        ...         "min_style": "fire",
        ...         "start_pressure": 0,
        ...         "end_pressure": 0,
        ...     }
        ... )
    """

    settings: LammpsSettings | dict | None = field(default=None)

    def __post_init__(self) -> None:
        """Initialize minimization-specific settings and defaults."""
        self.calc_type = "lammps_minimization"
        # Initialize settings if None
        if self.settings is None:
            settings_dict = {}
        elif isinstance(self.settings, LammpsSettings):
            settings_dict = self.settings.as_dict()
        else:
            settings_dict = self.settings.copy()

        # Add ensemble-specific defaults, using values from settings if provided
        settings_dict.update(
            {
                "ensemble": "minimize",
                "nsteps": settings_dict.get("nsteps", 10000),
                "start_pressure": settings_dict.get("start_pressure", 0),
                "end_pressure": settings_dict.get("end_pressure", 0),
                "tol": settings_dict.get("tol", 1.0e-6),
                "thermo": settings_dict.get(
                    "thermo", 5
                ),  # Use 5 for minimization like reference
                "traj_interval": settings_dict.get(
                    "traj_interval", 5
                ),  # Use 5 for minimization like reference
            }
        )

        # Set the updated dict back to self.settings
        self.settings = settings_dict
        super().__post_init__()
