"""Sets for the MPMorph amorphous workflows."""

from __future__ import annotations

from dataclasses import dataclass

from pymatgen.io.vasp.sets import MPMDSet

from atomate2.vasp.sets.core import MDSetGenerator


@dataclass
class MPMorphMDSetGenerator(MPMDSet):
    """
    Class to generate VASP molecular dynamics input sets for amorphous materials.

    This class wraps around pymatgen's `.MPMDSet` by adding sensible
    ensemble defaults from atomate2's `.MDSetGenerator`.

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
    spin_polarized
        Whether to do spin polarized calculations.
        The VASP ISPIN parameter. Defaults to False.
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
    spin_polarized : bool = False
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
        updates.update({
            "LAECHG": False,
            "EDIFFG": None,
            **MDSetGenerator._get_ensemble_defaults(self.structure, self.ensemble)
        })
        if self.spin_polarized:
            updates.update(MAGMOM = None)

        return updates