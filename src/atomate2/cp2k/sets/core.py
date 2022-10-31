"""Module defining core CP2K input set generators."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.cp2k.inputs import Cp2kInput
from pymatgen.io.cp2k.outputs import Cp2kOutput
from pymatgen.io.cp2k.utils import get_truncated_coulomb_cutoff

from atomate2.common.schemas.math import Vector3D
from atomate2.cp2k.sets.base import Cp2kInputGenerator, multiple_input_updators

logger = logging.getLogger(__name__)


__all__ = [
    "StaticSetGenerator",
    "RelaxSetGenerator",
    "CellOptSetGenerator",
    "HybridStaticSetGenerator",
    "HybridRelaxSetGenerator",
    "HybridCellOptSetGenerator",
    "NonSCFSetGenerator",
]


@dataclass
class StaticSetGenerator(Cp2kInputGenerator):
    """
    Class to generate CP2K static input sets.

    Parameters
    ----------

    """

    def get_input_updates(self, *args, **kwargs) -> dict:
        """
        Get updates to the input for a static CP2K job.

        Parameters
        ----------


        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        updates = {"run_type": "ENERGY_FORCE"}
        return updates

@dataclass
class RelaxSetGenerator(Cp2kInputGenerator):
    """

    """

    def get_input_updates(self, *args, **kwargs) -> dict:
        """
        """
        updates = {
            "run_type": "GEO_OPT",
            "activate_motion": {
                'optimizer': "BFGS",
                "trust_radius": 0.1
            },
            "modify_dft_print_iters": {"iters": 0, "add_last": "numeric"},
        }
        return updates

@dataclass
class CellOptSetGenerator(Cp2kInputGenerator):
    """

    """


    def get_input_updates(self, *args, **kwargs) -> dict:
        """
        """
        updates = {
            "run_type": "CELL_OPT",
            "activate_motion": {
                'optimizer': "BFGS",
                "trust_radius": 0.1
            },
            "modify_dft_print_iters": {"iters": 0, "add_last": "numeric"},
        }

        return updates


@dataclass
class HybridSetGenerator(Cp2kInputGenerator):

    hybrid_functional: str = "PBE0"

    def get_input_updates(self, structure, *args, **kwargs) -> dict:
        updates = {
            "activate_hybrid": {
                "hybrid_functional": self.hybrid_functional,
                "screen_on_initial_p": False,
                "screen_p_forces": False,
                "eps_schwarz": 1e-7,
                "eps_schwarz_forces": 1e-7,
            },
        }
        if hasattr(structure, "lattice"):
            updates['activate_hybrid']['cutoff_radius'] = get_truncated_coulomb_cutoff(structure)
        return updates

@dataclass
@multiple_input_updators()
class HybridStaticSetGenerator(HybridSetGenerator, StaticSetGenerator):
    pass

@dataclass
@multiple_input_updators()
class HybridRelaxSetGenerator(HybridSetGenerator, RelaxSetGenerator):
    pass

@dataclass
@multiple_input_updators()
class HybridCellOptSetGenerator(HybridSetGenerator, CellOptSetGenerator):
    pass

@dataclass
class NonSCFSetGenerator(Cp2kInputGenerator):
    """
    Class to generate CP2K non-self-consistent field input sets.

    **Note** cp2k doesn't have a true non scf option. All you can do is set
    max_scf to 1, and use a pre-converged wavefunction. While this seems to 
    be the same, it means that the kpoint grid used to generate the restart file
    needs to be present in the input set or the first scf step can slightly jump
    away from the minimum that was found.

    Parameters
    ----------
    mode
        Type of band structure mode. Options are "line", "uniform"
    reciprocal_density
        Density of k-mesh by reciprocal volume.
    line_density
        Line density for line mode band structure.
    """

    mode: str = "line"
    reciprocal_density: float = 100
    line_density: float = 20

    def __post_init__(self):
        """Ensure mode is set correctly."""
        super().__post_init__()
        self.mode = self.mode.lower()

        supported_modes = ("line", "uniform")
        if self.mode not in supported_modes:
            raise ValueError(f"Supported modes are: {', '.join(supported_modes)}")

    def get_kpoints_updates(
        self,
        structure: Structure,
        prev_input: Cp2kInput = None,
    ) -> dict:
        """
        Get updates to the kpoints configuration for a non-self consistent VASP job.

        Note, these updates will be ignored if the user has set user_kpoint_settings.

        Parameters
        ----------
        structure
            A structure.
        prev_incar
            An incar from a previous calculation.
        bandgap
            The band gap.
        vasprun
            A vasprun from a previous calculation.
        outcar
            An outcar from a previous calculation.

        Returns
        -------
        dict
            A dictionary of updates to apply to the KPOINTS config.
        """
        if self.mode == "line":
            return {"line_density": self.line_density}

        return {"reciprocal_density": self.reciprocal_density}

    def get_input_updates(
        self,
        structure: Structure,
        prev_input: Cp2kInput = None,
        cp2k_output: Cp2kOutput = None,
    ) -> dict:
        """
        """

        updates = {
            'max_scf': 1,
            'print_bandstructure': True, 
            'kpoints_line_density': self.line_density if self.mode == "line" else 1,
            'print_dos': True,
            'print_pdos': False, # Not possible as of 2022.1
            'print_mo_cubes': False,
            'run_type': 'ENERGY_FORCE',
        }

        return updates

@dataclass
class MDSetGenerator(Cp2kInputGenerator):
    """
    Class to generate VASP molecular dynamics input sets.

    Parameters
    ----------
    ensemble
        Molecular dynamics ensemble to run. All options are (from manual):
            HYDROSTATICSHOCK, ISOKIN, LANGEVIN, MSST, MSST_DAMPED, NPE_F, NPE_I,
            NPT_F, NPT_I, NPT_IA, NVE, NVT, NVT_ADIABATIC, REFTRAJ
    start_temp
        Starting temperature. TEMPERATURE
    end_temp
        Final temperature. The VASP `TEEND` parameter.
    nsteps
        Number of time steps for simulations. The VASP `NSW` parameter.
    time_step
        The time step (in femtosecond) for the simulation. The VASP `POTIM` parameter.
    **kwargs
        Other keyword arguments that will be passed to :obj:`VaspInputGenerator`.
    """

    ensemble: str = "NVT"
    temperature: float = 300
    nsteps: int = 1000
    time_step: int = 2
    thermostat: str = "NOSE"

    def get_input_updates(
        self,
        structure: Structure,
        prev_input: Cp2kInput = None,
        cp2k_output: Cp2kOutput = None,
    ) -> dict:
        """
        """
        updates = {
            "run_type": "MD",
            "activate_motion": {
                "ensemble": self.ensemble,
                "temperature": self.temperature,
                "timestep": self.time_step,
                "nsteps": self.nsteps,
                "thermostat": self.thermostat
            },
            "print_bandstructure": False, # Disable printing
            "print_dos": False,
            "print_pdos": False,
            "print_v_hartree": False,
            "print_e_density": False,
            "print_mo_cubes": False, 
        }

        return updates