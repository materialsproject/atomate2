"""Module defining ApproxNEB input set generators."""

from dataclasses import dataclass

from atomate2.vasp.sets.base import VaspInputGenerator


@dataclass
class ApproxNEBSetGenerator(VaspInputGenerator):
    """
    Class to generate VASP ApproxNEB input sets.

    Parameters
    ----------
    set_type: str
        Can be either "host" or "image", for different stages of relaxation
    """

    set_type: str = "host"

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
