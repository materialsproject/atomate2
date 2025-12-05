"""Schema definitions for force field tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from emmet.core.types.enums import StoreTrajectoryOption
from pydantic import BaseModel, Field
from pymatgen.core import Molecule

from atomate2.ase.schemas import (
    AseMoleculeTaskDoc,
    AseObject,
    AseResult,
    AseStructureTaskDoc,
    AseTaskDoc,
    _task_doc_translation_keys,
)
from atomate2.forcefields import MLFF

if TYPE_CHECKING:
    from typing_extensions import Self


class ForceFieldMeta(BaseModel):
    """Add metadata to forcefield output documents."""

    forcefield_name: str | None = Field(
        None,
        description="name of the interatomic potential used for relaxation.",
    )

    forcefield_version: str | None = Field(
        "Unknown",
        description="version of the interatomic potential used for relaxation.",
    )

    dir_name: str | None = Field(
        None, description="Directory where the force field calculations are performed."
    )

    included_objects: list[AseObject] | None = Field(
        None, description="list of forcefield objects included with this task document"
    )
    objects: dict[AseObject, Any] | None = Field(
        None, description="Forcefield objects associated with this task"
    )

    is_force_converged: bool | None = Field(
        None,
        description=(
            "Whether the calculation is converged with respect to interatomic forces."
        ),
    )

    @property
    def forcefield_objects(self) -> dict[AseObject, Any] | None:
        """Alias `objects` attr for backwards compatibility."""
        return self.objects


class ForceFieldMoleculeTaskDocument(AseMoleculeTaskDoc, ForceFieldMeta):
    """Document containing information on molecule manipulation using a force field."""

    @classmethod
    def from_ase_task_doc(
        cls, ase_task_doc: AseTaskDoc, **task_document_kwargs
    ) -> Self:
        """Create a ForceFieldMoleculeTaskDocument from an AseTaskDoc.

        Parameters
        ----------
        ase_task_doc : AseTaskDoc
            Task doc for the calculation
        task_document_kwargs : dict
            Additional keyword args passed to :obj:`.AseStructureTaskDoc()`.
        """
        task_document_kwargs.update(
            {k: getattr(ase_task_doc, k) for k in _task_doc_translation_keys},
            structure=ase_task_doc.mol_or_struct,
        )
        return cls.from_molecule(
            meta_molecule=ase_task_doc.mol_or_struct, **task_document_kwargs
        )


class ForceFieldTaskDocument(AseStructureTaskDoc, ForceFieldMeta):
    """Document containing information on atomistic manipulation using a force field."""

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
    ) -> Self | ForceFieldMoleculeTaskDocument:
        """Create forcefield output for a task that has ASE-compatible outputs.

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
            MLFF.MACE_MP_0: "mace-torch",
            MLFF.MACE_MPA_0: "mace-torch",
            MLFF.MACE_MP_0B3: "mace-torch",
            MLFF.GAP: "quippy-ase",
            MLFF.Nequip: "nequip",
            MLFF.DeepMD: "deepmd-kit",
            MLFF.MATPES_PBE: "matgl",
            MLFF.MATPES_R2SCAN: "matgl",
        }

        if pkg_name := {str(k): v for k, v in model_to_pkg_map.items()}.get(
            ff_kwargs["forcefield_name"]
        ):
            import importlib.metadata

            ff_kwargs["forcefield_version"] = importlib.metadata.version(pkg_name)

        return (
            ForceFieldMoleculeTaskDocument
            if isinstance(result.final_mol_or_struct, Molecule)
            else cls
        ).from_ase_task_doc(ase_task_doc, **ff_kwargs)
