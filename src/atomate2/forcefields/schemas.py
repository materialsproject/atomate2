"""Schema definitions for force field tasks."""

from __future__ import annotations

from typing import Any, Optional

from emmet.core.utils import ValueEnum
from emmet.core.vasp.calculation import StoreTrajectoryOption
from monty.dev import deprecated
from pydantic import Field
from pymatgen.core import Structure

from atomate2.ase.schemas import AseObject, AseResult, AseStructureTaskDoc, AseTaskDoc
from atomate2.forcefields import MLFF


@deprecated(replacement=AseResult, deadline=(2025, 1, 1))
class ForcefieldResult(AseResult):
    """Schema to store outputs; deprecated."""

    final_structure: Optional[Structure] = Field(
        None, description="The structure in the final trajectory frame."
    )

    def model_post_init(self, __context: Any) -> None:
        """Populate final_structure attr."""
        self.final_structure = getattr(
            self, "final_structure", self.final_mol_or_struct
        )


@deprecated(replacement=AseObject, deadline=(2025, 1, 1))
class ForcefieldObject(ValueEnum):
    """Types of force-field output data objects."""

    TRAJECTORY = "trajectory"


class ForceFieldTaskDocument(AseStructureTaskDoc):
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

    @classmethod
    def from_ase_compatible_result(
        cls,
        ase_calculator_name: str,
        result: AseResult,
        steps: int,
        relax_kwargs: dict = None,
        optimizer_kwargs: dict = None,
        fix_symmetry: bool = False,
        symprec: float = 1e-2,
        ionic_step_data: tuple = (
            "energy",
            "forces",
            "magmoms",
            "stress",
            "mol_or_struct",
        ),
        store_trajectory: StoreTrajectoryOption = StoreTrajectoryOption.NO,
        tags: list[str] | None = None,
        **task_document_kwargs,
    ) -> ForceFieldTaskDocument:
        """Create an AseTaskDoc for a task that has ASE-compatible outputs.

        Parameters
        ----------
        ase_calculator_name : str
            Name of the ASE calculator used.
        result : AseResult
            The output results from the task.
        fix_symmetry : bool
            Whether to fix the symmetry of the ions during relaxation.
        symprec : float
            Tolerance for symmetry finding in case of fix_symmetry.
        steps : int
            Maximum number of ionic steps allowed during relaxation.
        relax_kwargs : dict
            Keyword arguments that will get passed to :obj:`Relaxer.relax`.
        optimizer_kwargs : dict
            Keyword arguments that will get passed to :obj:`Relaxer()`.
        ionic_step_data : tuple
            Which data to save from each ionic step.
        store_trajectory:
            whether to set the StoreTrajectoryOption
        tags : list[str] or None
            A list of tags for the task.
        task_document_kwargs : dict
            Additional keyword args passed to :obj:`.AseTaskDoc()`.
        """
        ase_task_doc = AseTaskDoc.from_ase_compatible_result(
            ase_calculator_name=ase_calculator_name,
            result=result,
            fix_symmetry=fix_symmetry,
            symprec=symprec,
            steps=steps,
            relax_kwargs=relax_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            ionic_step_data=ionic_step_data,
            store_trajectory=store_trajectory,
            tags=tags,
            **task_document_kwargs,
        )
        ff_kwargs = {
            "forcefield_name": task_document_kwargs.get(
                "forcefield_name", ase_calculator_name
            )
        }

        # map force field name to its package name
        model_to_pkg_map = {
            MLFF.M3GNet: "matgl",
            MLFF.CHGNet: "chgnet",
            MLFF.MACE: "mace-torch",
            MLFF.GAP: "quippy-ase",
            MLFF.Nequip: "nequip",
        }

        if pkg_name := {str(k): v for k, v in model_to_pkg_map.items()}.get(
            ff_kwargs["forcefield_name"]
        ):
            import importlib.metadata

            ff_kwargs["forcefield_version"] = importlib.metadata.version(pkg_name)

        return cls.from_ase_task_doc(ase_task_doc, **ff_kwargs)

    @property
    def forcefield_objects(self) -> Optional[dict[AseObject, Any]]:
        """Alias `objects` attr for backwards compatibility."""
        return self.objects
