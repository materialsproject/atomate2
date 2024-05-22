"""Sets for the MPMorph amorphous workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from atomate2.vasp.sets.core import MDSetGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun


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
    inherit_incar: bool | None = False
    ensemble: str = "nvt"
    time_step: float = 2
    nsteps: int = 2000

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for a molecular dynamics job.

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
            A dictionary of updates to apply.
        """
        updates = super().get_incar_updates(
            structure=structure,
            prev_incar=prev_incar,
            bandgap=bandgap,
            vasprun=vasprun,
            outcar=outcar,
        )
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
            }
        )

        return updates

    def get_kpoints_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0.0,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
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
        return {"gamma_only": True}
