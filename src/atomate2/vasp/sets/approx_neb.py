"""Module defining ApproxNEB input set generators."""

from dataclasses import dataclass
from typing import Literal

from pymatgen.io.vasp.sets import MPRelaxSet


@dataclass
class ApproxNEBSetGenerator(MPRelaxSet):
    """
    Class to generate VASP ApproxNEB input sets.

    Parameters
    ----------
    set_type: str
        Can be either "host" or "image", for different stages of relaxation
    """

    auto_ismear: bool = False
    auto_kspacing: bool = False
    inherit_incar: bool | None = False
    bandgap_tol: float = None
    force_gamma: bool = True
    auto_metal_kpoints: bool = True
    symprec: float | None = None
    set_type: Literal["image", "host"] = "host"

    def __post_init__(self) -> None:
        """Ensure correct settings for class attrs."""
        super().__post_init__()
        if self.set_type not in {"image", "host"}:
            raise ValueError(
                f'Unrecognized {self.set_type=}; must be "image" or "host".'
            )

    @property
    def incar_updates(self) -> dict:
        """
        Get updates to the INCAR settings for an ApproxNEB job.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        updates = {
            "EDIFF": 1e-4,
            "EDIFFG": -0.05,
            "IBRION": 1,
            "ISIF": 3,
            "ISMEAR": 0,
            "LDAU": False,
            "NSW": 400,
            "ADDGRID": True,
            "ISYM": 1,
            "NELMIN": 4,
        }

        if self.set_type == "image":
            updates.update({"ISIF": 2, "ISYM": 0})

        return updates
