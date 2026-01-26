"""Schema definitions for force field tasks."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from emmet.core.types.enums import StoreTrajectoryOption
from pydantic import BaseModel, Field
from pymatgen.core import Molecule
from typing_extensions import assert_never

from atomate2.ase.schemas import (
    AseMoleculeTaskDoc,
    AseObject,
    AseResult,
    AseStructureTaskDoc,
    AseTaskDoc,
    _task_doc_translation_keys,
)
from atomate2.forcefields import MLFF
from atomate2.forcefields.utils import _get_standardized_mlff, _load_calc_cls

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
        calculator_meta: MLFF | dict | None = None,
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
        calculator_meta : Optional, MLFF or dict or None
            Metadata about the calculator used.
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
            ),
        }

        # Infer `calculator_meta` for MLFFs if not provided
        if (calculator_meta is None) and ase_calculator_name.startswith("MLFF."):
            calculator_meta = _get_standardized_mlff(ase_calculator_name)
        # Populate forcefield version if possible
        if calculator_meta is None:
            warnings.warn(
                "Could not determine forcefield version as calculator_meta was not "
                "provided.",
                stacklevel=2,
            )
        elif pkg_name := _get_pkg_name(calculator_meta):
            from importlib.metadata import PackageNotFoundError, version

            try:
                ff_kwargs["forcefield_version"] = version(pkg_name)
            except PackageNotFoundError:
                # In cases where the package name (`mace_torch`) is not the same
                # as the import string
                from importlib import import_module

                ff_kwargs["forcefield_version"] = getattr(
                    import_module(pkg_name), "__version__", None
                )

        return (
            ForceFieldMoleculeTaskDocument
            if isinstance(result.final_mol_or_struct, Molecule)
            else cls
        ).from_ase_task_doc(ase_task_doc, **ff_kwargs)


def _get_pkg_name(calculator_meta: MLFF | dict) -> str | None:
    """Get the package name for a given force field."""
    if isinstance(calculator_meta, MLFF):
        # map force field name to its package name
        ff_pkg = None
        match calculator_meta:
            case MLFF.M3GNet | MLFF.CHGNet | MLFF.MATPES_PBE | MLFF.MATPES_R2SCAN:
                ff_pkg = "matgl"
            case MLFF.MACE | MLFF.MACE_MP_0 | MLFF.MACE_MPA_0 | MLFF.MACE_MP_0B3:
                ff_pkg = "mace-torch"
            case MLFF.GAP:
                ff_pkg = "quippy-ase"
            case MLFF.Nequip:
                ff_pkg = "nequip"
            case MLFF.DeepMD:
                ff_pkg = "deepmd-kit"
        return ff_pkg
    if isinstance(calculator_meta, dict):
        calc_cls = _load_calc_cls(calculator_meta)
        return calc_cls.__module__.split(".", 1)[0]
    assert_never(calculator_meta)
