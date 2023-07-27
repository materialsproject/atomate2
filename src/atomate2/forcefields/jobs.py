"""Job to relax a structure using a force field (aka an interatomic potential)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Maker, job

from atomate2.forcefields.schemas import ForceFieldTaskDocument

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)

@dataclass
class ForceFieldRelaxMaker(Maker):
    """
    Base Maker to calculate forces and stresses using any force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the forcefield.
    relax_cell : bool
        Whether to allow the cell shape/volume to change during relaxation.
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """
    name: str = "Forcefield relax"
    force_field_name: str = "Forcefield"
    relax_cell: bool = False
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    @job(output_schema=ForceFieldTaskDocument)
    def make(self, structure: Structure):
        if self.steps < 0:
            logger.warning(
                "WARNING: A negative number of steps is not possible. "
                "Behavior may vary..."
            )


        result = self._relax(structure, self.relax_cell, self.steps, self.relax_kwargs, self.optimizer_kwargs)

        return ForceFieldTaskDocument.from_ase_compatible_result(
            self.force_field_name,
            result,
            self.relax_cell,
            self.steps,
            self.relax_kwargs,
            self.optimizer_kwargs,
            **self.task_document_kwargs,
        )

    def _relax(self, structure, relax_cell, steps, relax_kwargs, optimizer_kwargs):
        raise NotImplementedError


@dataclass
class ForceFieldStaticMaker(ForceFieldRelaxMaker):
    """
    Maker to calculate forces and stresses using the CHGNet force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the force field.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = "ForceField static"
    force_field_name : str = "Forcefield"
    task_document_kwargs: dict = field(default_factory=dict)

    @job(output_schema=ForceFieldTaskDocument)
    def make(self, structure: Structure):
        if self.steps < 0:
            logger.warning(
                "WARNING: A negative number of steps is not possible. "
                "Behavior may vary..."
            )

        result = self._evaluate_static(structure)

        return ForceFieldTaskDocument.from_ase_compatible_result(
            self.force_field_name,
            result,
            self.relax_cell,
            self.steps,
            self.relax_kwargs,
            self.optimizer_kwargs,
            **self.task_document_kwargs,
        )

    def _evaluate_static(self, structure):
        raise NotImplementedError

@dataclass
class CHGNetRelaxMaker(ForceFieldRelaxMaker):
    """
    Maker to perform a relaxation using the CHGNet universal ML force field.

    Parameters
    ----------
    force_field_name : str
        The name of the forcefield.
    relax_cell : bool
        Whether to allow the cell shape/volume to change during relaxation.
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = "CHGNet relax"
    force_field_name = "CHGNet"
    relax_cell: bool = False
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)


    def _relax(self, structure, relax_cell, steps, relax_kwargs, optimizer_kwargs):
        from chgnet.model import StructOptimizer
        relaxer = StructOptimizer(**optimizer_kwargs)
        result = relaxer.relax(
            structure, relax_cell=relax_cell, steps=steps, **relax_kwargs
        )
        return result


@dataclass
class CHGNetStaticMaker(ForceFieldStaticMaker):
    """
    Maker to calculate forces and stresses using the CHGNet force field.

    Parameters
    ----------
    name : str
        The job name.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = "CHGNet static"
    task_document_kwargs: dict = field(default_factory=dict)
    def _evaluate_static(self, structure):
        from chgnet.model import StructOptimizer
        relaxer = StructOptimizer()
        result = relaxer.relax(
            structure, steps=1
        )
        return result

@dataclass
class M3GNetRelaxMaker(Maker):
    """
    Maker to perform a relaxation using the M3GNet universal ML force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the forcefield.
    relax_cell : bool
        Whether to allow the cell shape/volume to change during relaxation.
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = "M3GNet relax"
    force_field_name: str = "M3GNet"
    relax_cell: bool = False
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    def _relax(self, structure, relax_cell, steps, relax_kwargs, optimizer_kwargs):
        import matgl
        from matgl.ext.ase import Relaxer

        # Note: the below code was taken from the matgl repo examples.
        # Load pre-trained M3GNet model (currently uses the MP-2021.2.8 database)
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        relaxer = Relaxer(
            potential=pot,
            relax_cell=relax_cell,
            **optimizer_kwargs,
        )

        result = relaxer.relax(
            structure,
            steps=steps,
            **relax_kwargs,
        )
        return result


@dataclass
class M3GNetStaticMaker(M3GNetRelaxMaker):
    """
    Maker to calculate forces and stresses using the M3GNet force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the forcefield.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = "M3GNet static"
    force_field_name: str = "M3GNet"
    task_document_kwargs: dict = field(default_factory=dict)

    def _evaluate_static(self, structure):
        import matgl
        from matgl.ext.ase import Relaxer

        # Note: the below code was taken from the matgl repo examples.
        # Load pre-trained M3GNet model (currently uses the MP-2021.2.8 database)
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        relaxer = Relaxer(
            potential=pot,
            relax_cell=False,

        )

        result = relaxer.relax(
            structure,
            steps=1,
        )
        return result