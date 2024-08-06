"""Module defining core CP2K input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pymatgen.io.cp2k.utils import get_truncated_coulomb_cutoff

from atomate2.cp2k.sets.base import Cp2kInputGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.cp2k.inputs import Cp2kInput
    from pymatgen.io.cp2k.outputs import Cp2kOutput

logger = logging.getLogger(__name__)


@dataclass
class StaticSetGenerator(Cp2kInputGenerator):
    """Class to generate CP2K static input sets."""

    def get_input_updates(self, *args, **kwargs) -> dict:
        """Get updates to the input for a static job."""
        return {"run_type": "ENERGY_FORCE"}


@dataclass
class RelaxSetGenerator(Cp2kInputGenerator):
    """
    Class to generate CP2K relax sets.

    I.e., sets for optimization of internal coordinates without cell parameter
    optimization.
    """

    def get_input_updates(self, *args, **kwargs) -> dict:
        """Get updates to the input for a relax job."""
        return {
            "run_type": "GEO_OPT",
            "activate_motion": {"optimizer": "BFGS", "trust_radius": 0.1},
        }


@dataclass
class CellOptSetGenerator(Cp2kInputGenerator):
    """
    Class to generate CP2K cell optimization sets.

    I.e., sets for optimization of both internal coordinates and the lattice vectors.
    """

    def get_input_updates(self, *args, **kwargs) -> dict:
        """Get updates to the input for a cell opt job."""
        return {
            "run_type": "CELL_OPT",
            "activate_motion": {"optimizer": "BFGS", "trust_radius": 0.1},
        }


@dataclass
class HybridStaticSetGenerator(Cp2kInputGenerator):
    """Class for generating static hybrid input sets."""

    def get_input_updates(self, structure: Structure, *args, **kwargs) -> dict:
        """Get input updates for a hybrid calculation."""
        updates: dict = {
            "run_type": "ENERGY_FORCE",
            "activate_hybrid": {
                "hybrid_functional": "PBE0",
                "screen_on_initial_p": False,
                "screen_p_forces": False,
                "eps_schwarz": 1e-7,
                "eps_schwarz_forces": 1e-5,
            },
        }
        if hasattr(structure, "lattice"):
            updates["activate_hybrid"]["cutoff_radius"] = get_truncated_coulomb_cutoff(
                structure
            )

        return updates


@dataclass
class HybridRelaxSetGenerator(Cp2kInputGenerator):
    """Class for generating hybrid relaxation input sets."""

    def get_input_updates(self, structure: Structure, *args, **kwargs) -> dict:
        """Get input updates for a hybrid calculation."""
        updates: dict = {
            "run_type": "GEO_OPT",
            "activate_motion": {"optimizer": "BFGS", "trust_radius": 0.1},
            "activate_hybrid": {
                "hybrid_functional": "PBE0",
                "screen_on_initial_p": False,
                "screen_p_forces": False,
                "eps_schwarz": 1e-7,
                "eps_schwarz_forces": 1e-5,
            },
        }
        if hasattr(structure, "lattice"):
            updates["activate_hybrid"]["cutoff_radius"] = get_truncated_coulomb_cutoff(
                structure
            )

        return updates


@dataclass
class HybridCellOptSetGenerator(Cp2kInputGenerator):
    """Class for generating hybrid cell optimization input sets."""

    def get_input_updates(self, structure: Structure, *args, **kwargs) -> dict:
        """Get input updates for a hybrid calculation."""
        updates: dict = {
            "run_type": "CELL_OPT",
            "activate_motion": {"optimizer": "BFGS", "trust_radius": 0.1},
            "activate_hybrid": {
                "hybrid_functional": "PBE0",
                "screen_on_initial_p": False,
                "screen_p_forces": False,
                "eps_schwarz": 1e-7,
                "eps_schwarz_forces": 1e-5,
            },
        }
        if hasattr(structure, "lattice"):
            updates["activate_hybrid"]["cutoff_radius"] = get_truncated_coulomb_cutoff(
                structure
            )

        return updates


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

    def __post_init__(self) -> None:
        """Ensure mode is set correctly."""
        self.mode = self.mode.lower()

        supported_modes = ("line", "uniform")
        if self.mode not in supported_modes:
            raise ValueError(f"Supported modes are: {', '.join(supported_modes)}")

    def get_kpoints_updates(
        self,
        structure: Structure,
        prev_input: Cp2kInput = None,
    ) -> dict:
        """Get updates to the kpoints configuration for a non-self consistent VASP job.

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
        """Get input updates for a non scf calculation."""
        return {
            "max_scf": 1,
            "print_bandstructure": True,
            "kpoints_line_density": self.line_density if self.mode == "line" else 1,
            "print_dos": True,
            "print_pdos": False,  # Not possible as of 2022.1
            "print_mo_cubes": False,
            "run_type": "ENERGY_FORCE",
        }


@dataclass
class MDSetGenerator(Cp2kInputGenerator):
    """Class to generate molecular dynamics input sets."""

    def get_input_updates(self, structure: Structure, *args, **kwargs) -> dict:
        """Get input updates for running a MD calculation."""
        return {
            "run_type": "MD",
            "activate_motion": {
                "ensemble": "NVT",
                "temperature": 300,
                "timestep": 2,
                "nsteps": 1000,
                "thermostat": "NOSE",
            },
            "print_bandstructure": False,  # Disable printing
            "print_dos": False,
            "print_pdos": False,
            "print_v_hartree": False,
            "print_e_density": False,
            "print_mo_cubes": False,
        }
