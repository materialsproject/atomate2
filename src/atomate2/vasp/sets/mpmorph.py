"""Sets for the MPMorph amorphous workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from atomate2.vasp.sets.core import MDSetGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun

from pymatgen.io.vasp import Kpoints


@dataclass
class MPMorphMDSetGenerator(MDSetGenerator):
    """
    Class to generate VASP molecular dynamics input sets for amorphous materials.

    Parameters
    ----------
    ensemble
        Molecular dynamics ensemble to run. Options include `nvt`, `nve`, and `npt`.
    start_temp
        Starting temperature. The VASP `TEBEG` parameter.
    end_temp
        Final temperature. The VASP `TEEND` parameter.
    nsteps
        Number of time steps for simulations. The VASP `NSW` parameter.
    time_step
        The time step (in femtosecond) for the simulation. The VASP `POTIM` parameter.
    **kwargs
        Other keyword arguments that will be passed to :obj:`VaspInputGenerator`.
    """

    auto_ismear: bool = False
    auto_kspacing: bool = True
    auto_ispin: bool = False
    auto_lreal: bool = False
    inherit_incar: bool | None = False
    ensemble: str = "nvt"
    time_step: float = 2
    nsteps: int = 2000

    @property
    def incar_updates(self) -> dict:
        """
        Get updates to the INCAR for a molecular dynamics job.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        updates = super().incar_updates
        updates.update(
            {
                "ISPIN": 1,  # Do not consider magnetism in AIMD simulations
                # Perform calculation in real space for AIMD due to large unit cell size
                "LREAL": "Auto",
                "LAECHG": False,  # Don't need AECCAR for AIMD
                "EDIFFG": None,  # Does not apply to MD simulations, see: https://www.vasp.at/wiki/index.php/EDIFFG
                "GGA": "PS",
                "LPLANE": False,  # LPLANE is recommended to be False on Cray machines (https://www.vasp.at/wiki/index.php/LPLANE)
                "LDAUPRINT": 0,
                "MAGMOM": None,  # Compatability with non-spin polarized calculations
            }
        )

        return updates

    @property
    def kpoints_updates(self) -> dict:
        """
        Get updates to the kpoints configuration for a non-self consistent VASP job.

        Note, these updates will be ignored if the user has set user_kpoint_settings.

        Returns
        -------
        dict
            A dictionary of updates to apply to the KPOINTS config.
        """
        return Kpoints()
