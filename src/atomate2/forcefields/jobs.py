"""Job to relax a structure using a force field (aka an interatomic potential)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from jobflow import Maker, job
from pymatgen.core.structure import Structure

from atomate2.forcefields.schemas import FFStructureRelaxDocument

logger = logging.getLogger(__name__)


@dataclass
class CHGNetRelaxMaker(Maker):
    """
    Maker to perform a relaxation using the CHGNet universal machine learning force field. 

    Parameters
    ----------
    name : str
        The job name.
    relax_cell : bool
        Whether to allow the cell shape/volume to change during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`StructOptimizer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`StructOptimizer()`.
    keep_info : list
        Which information from the relaxation trajectory to save using
        the :obj:`.FFStructureRelaxDocument.from_chgnet_result`.

    """

    name: str = "CHGNet relax"
    relax_cell: bool = False
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    keep_info: list = field(
        default_factory=lambda: [
            "energies",
            "forces",
            "stresses",
            "magmoms",
            "atom_positions",
            "cells",
        ]
    )
    # NOTE: the 'atoms' field will always be removed later because it is not
    # serializable (as of May 2023)

    @job(output_schema=FFStructureRelaxDocument)
    def make(self, structure: Structure):
        """
        Perform a relaxation of a structure using CHGNet.

        Parameters
        ----------
        structure: ~pymatgen.core.structure.Structure
            A pymatgen structure.

        """
        from chgnet.model import StructOptimizer

        relaxer = StructOptimizer(**self.optimizer_kwargs)
        result = relaxer.relax(
            structure, relax_cell=self.relax_cell, **self.relax_kwargs
        )

        ff_structure_relax_doc = FFStructureRelaxDocument.from_chgnet_result(
            structure,
            self.relax_cell,
            self.relax_kwargs,
            self.optimizer_kwargs,
            result,
            self.keep_info,
        )

        return ff_structure_relax_doc
