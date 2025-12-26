"""Job to relax a structure using a force field (aka an interatomic potential)."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import job

from atomate2.ase.jobs import AseRelaxMaker
from atomate2.forcefields.schemas import ForceFieldTaskDocument
from atomate2.forcefields.utils import _FORCEFIELD_DATA_OBJECTS, MLFF, ForceFieldMixin

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pymatgen.core.structure import Molecule, Structure

    from atomate2.forcefields.schemas import ForceFieldMoleculeTaskDocument

logger = logging.getLogger(__name__)


def forcefield_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of forcefield job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all forcefield jobs. For example, it ensures that large data objects
    (currently only trajectories) are all stored in the atomate2 data store.
    It also configures the output schema to be a
    ForceFieldTaskDocument :obj:`.TaskDoc`. or
    ForceFieldMoleculeTaskDocument :obj:`.TaskDoc`.

    Any makers that return forcefield jobs (not flows) should decorate the
    ``make`` method with @forcefield_job. For example:

    .. code-block:: python

        class MyForcefieldMaker(Maker):
            @forcefield_job
            def make(structure):
                # code to run forcefield job.
                pass

    Parameters
    ----------
    method : callable
        A Maker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate forcefield jobs.
    """
    return job(method, data=_FORCEFIELD_DATA_OBJECTS)


@dataclass
class ForceFieldRelaxMaker(ForceFieldMixin, AseRelaxMaker):
    """
    Base Maker to calculate forces and stresses using any force field.

    Should be subclassed to use a specific force field. By default,
    the code attempts to use the `self.force_field_name` attr to look
    up a predefined forcefield. To overwrite this behavior, redefine `self.calculator`.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF or dict
        The name of the force field.
    relax_cell : bool = True
        Whether to allow the cell shape/volume to change during relaxation.
    fix_symmetry : bool = False
        Whether to fix the symmetry during relaxation.
        Refines the symmetry of the initial structure.
    symprec : float | None = 1e-2
        Tolerance for symmetry finding in case of fix_symmetry.
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    ionic_step_data : tuple[str,...] or None
        Quantities to store in the TaskDocument ionic_steps.
        Possible options are "struct_or_mol", "energy",
        "forces", "stress", and "magmoms".
        "structure" and "molecule" are aliases for "struct_or_mol".
    store_trajectory : emmet .StoreTrajectoryOption = "no"
        Whether to store trajectory information ("no") or complete trajectories
        ("partial" or "full", which are identical).
    tags : list[str] or None
        A list of tags for the task.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()` or
          :obj: `ForceFieldMoleculeTaskDocument`.
    """

    name: str = "Force field relax"
    force_field_name: str | MLFF | dict = MLFF.Forcefield
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float | None = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    @forcefield_job
    def make(
        self, structure: Molecule | Structure, prev_dir: str | Path | None = None
    ) -> ForceFieldTaskDocument | ForceFieldMoleculeTaskDocument:
        """
        Perform a relaxation of a structure using a force field.

        Parameters
        ----------
        structure: .Structure or Molecule
            pymatgen structure or molecule.
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
                added to match the method signature of other makers.
        """
        ase_result = self._run_ase_safe(structure, prev_dir=prev_dir)

        if len(self.task_document_kwargs) > 0:
            warnings.warn(
                "`task_document_kwargs` is now deprecated, please use the top-level "
                "attributes `ionic_step_data` and `store_trajectory`",
                category=DeprecationWarning,
                stacklevel=1,
            )

        return ForceFieldTaskDocument.from_ase_compatible_result(
            self.ase_calculator_name,
            self.calculator_meta,
            ase_result,
            self.steps,
            relax_kwargs=self.relax_kwargs,
            optimizer_kwargs=self.optimizer_kwargs,
            relax_cell=self.relax_cell,
            fix_symmetry=self.fix_symmetry,
            symprec=self.symprec if self.fix_symmetry else None,
            ionic_step_data=self.ionic_step_data,
            store_trajectory=self.store_trajectory,
            tags=self.tags,
            **self.task_document_kwargs,
        )


@dataclass
class ForceFieldStaticMaker(ForceFieldRelaxMaker):
    """
    Maker to calculate forces and stresses using any force field.

    Note that while `steps = 1` by default, the user could override
    this setting along with cell shape relaxation (`relax_cell = False`
    by default).

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF or dict
        The name of the force field.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()` or
          :obj: `ForceFieldMoleculeTaskDocument`.
    """

    name: str = "Force field static"
    force_field_name: str | MLFF | dict = MLFF.Forcefield
    relax_cell: bool = False
    steps: int = 1
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
