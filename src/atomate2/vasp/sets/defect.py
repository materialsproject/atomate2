"""Module defining VASP input set generators for defect calculations."""

from dataclasses import dataclass

from pymatgen.core import Structure
from pymatgen.io.vasp import Outcar, Vasprun

from atomate2.vasp.sets.base import VaspInputSetGenerator


@dataclass
class AtomicRelaxSetGenerator(VaspInputSetGenerator):
    """Class to generate VASP atom-only relaxation input sets."""

    def __post_init__(self):
        """Initialize the class."""
        super().__post_init__()
        self.use_structure_charge = True

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for a relaxation job.

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
            "EDIFFG": -0.005,
            "LREAL": False,
            "NSW": 99,
            "LCHARG": False,
        }
