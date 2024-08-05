"""Schema definitions for force field tasks."""

from typing import Any, Optional

from emmet.core.utils import ValueEnum
from monty.dev import deprecated
from pydantic import Field

from atomate2.ase.schemas import AseObject, AseResult, AseTaskDocument
from atomate2.forcefields import MLFF


@deprecated(replacement=AseResult, deadline=(2025, 1, 1))
class ForcefieldResult(AseResult):
    """Schema to store outputs; deprecated."""


@deprecated(replacement=AseObject, deadline=(2025, 1, 1))
class ForcefieldObject(ValueEnum):
    """Types of force-field output data objects."""

    TRAJECTORY = "trajectory"


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
        if (self.forcefield_name is None) and (self.ase_calculator_name is not None):
            self.forcefield_name = self.ase_calculator_name

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

        if pkg_name := pkg_names.get(self.forcefield_name):
            import importlib.metadata

            self.forcefield_version = importlib.metadata.version(pkg_name)

    @property
    def forcefield_objects(self) -> dict[AseObject, Any] | None:
        """Alias `objects` attr for backwards compatibility."""
        return self.objects
