"""Job to relax a structure using a force field (aka an interatomic potential)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from jobflow import Maker, job
from pymatgen.core.structure import Structure

from atomate2.forcefields.schemas import ForceFieldTaskDocument

logger = logging.getLogger(__name__)


@dataclass
class CHGNetRelaxMaker(Maker):
    """
    Maker to perform a relaxation using the CHGNet universal ML force field.

    Parameters
    ----------
    name : str
        The job name.
    relax_cell : bool
        Whether to allow the cell shape/volume to change during relaxation.
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`StructOptimizer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`StructOptimizer()`.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = "CHGNet relax"
    relax_cell: bool = False
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    @job(output_schema=ForceFieldTaskDocument)
    def make(self, structure: Structure):
        """
        Perform a relaxation of a structure using CHGNet.

        Parameters
        ----------
        structure: .Structure
            A pymatgen structure.
        """
        from chgnet.model import StructOptimizer

        if self.steps < 0:
            logger.warning(
                "WARNING: A negative number of steps is not possible. "
                "Behavior may vary..."
            )

        relaxer = StructOptimizer(**self.optimizer_kwargs)
        result = relaxer.relax(
            structure, relax_cell=self.relax_cell, steps=self.steps, **self.relax_kwargs
        )

        ff_task_doc = ForceFieldTaskDocument.from_chgnet_result(
            result,
            self.relax_cell,
            self.steps,
            self.relax_kwargs,
            self.optimizer_kwargs,
            **self.task_document_kwargs,
        )

        return ff_task_doc


@dataclass
class CHGNetStaticMaker(CHGNetRelaxMaker):
    """
    Maker to calculate forces and stresses using the CHGNet force field.

    Parameters
    ----------
    name : str
        The job name.
    relax_cell : bool
        Whether to allow the cell shape/volume to change during relaxation.
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`StructOptimizer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`StructOptimizer()`.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = "CHGNet static"
    relax_cell: bool = False
    steps: int = 1
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
