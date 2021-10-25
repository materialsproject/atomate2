"""Module defining VASP input set generators for elastic tensor calculations."""

from atomate2.vasp.sets.base import VaspInputSetGenerator

__all__ = ["ElasticDeformationSetGenerator"]

from pymatgen.core import Structure
from pymatgen.io.vasp import Outcar, Vasprun


class ElasticDeformationSetGenerator(VaspInputSetGenerator):
    """
    Class to generate elastic deformation relaxation input sets.

    This input set is for a tight relaxation, where only the atomic positions are
    allowed to relax (ISIF=2). Both the k-point mesh density and convergence parameters
    are stricter than a normal relaxation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config_dict["KPOINTS"] = {"grid_density": 7000}

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for a VASP HSE06 band structure job.

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
        return {
            "IBRION": 2,
            "ISIF": 2,
            "ENCUT": 700,
            "EDIFF": 1e-7,
            "LAECHG": False,
            "EDIFFG": -0.001,
            "LREAL": False,
            "ALGO": "Normal",
            "NSW": 99,
            "LCHARG": False,
        }
