"""Schema definitions for force field tasks."""

from typing import Any, Optional

from pydantic import Field

from atomate2.ase.schemas import AseObject, AseTaskDocument
from atomate2.forcefields import MLFF


class ForceFieldTaskDocument(AseTaskDocument):
    """Document containing information on structure manipulation using a force field."""

    forcefield_name: Optional[str] = Field(
        None,
        description="name of the interatomic potential used for relaxation.",
    )

    forcefield_version: Optional[str] = Field(
        "Unknown",
        description="version of the interatomic potential used for relaxation.",
    )

    dir_name: Optional[str] = Field(
        None, description="Directory where the force field calculations are performed."
    )

    included_objects: Optional[list[AseObject]] = Field(
        None, description="list of forcefield objects included with this task document"
    )
    objects: Optional[dict[AseObject, Any]] = Field(
        None, description="Forcefield objects associated with this task"
    )

    is_force_converged: Optional[bool] = Field(
        None,
        description=(
            "Whether the calculation is converged with respect "
            "to interatomic forces."
        ),
    )

    def model_post_init(self, __context: Any) -> None:
        """Find forcefield version and name from defined attrs."""
        self.forcefield_name = getattr(self,"forcefield_name",self.ase_calculator_name)

        # map force field name to its package name
        pkg_names = {
            str(k): v
            for k, v in {
                MLFF.M3GNet: "matgl",
                MLFF.CHGNet: "chgnet",
                MLFF.MACE: "mace-torch",
                MLFF.GAP: "quippy-ase",
                MLFF.Nequip: "nequip",
            }.items()
        }

        if (pkg_name := pkg_names.get(self.forcefield_name)):
            import importlib.metadata

            self.forcefield_version = importlib.metadata.version(pkg_name)
