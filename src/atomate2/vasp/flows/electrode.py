"""Flow for electrode analysis with specific VASP implementations."""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from atomate2.common.flows import electrode as electrode_flows

if TYPE_CHECKING:
    from emmet.core.tasks import TaskDoc
    from pymatgen.io.vasp.outputs import VolumetricData

logger = logging.getLogger(__name__)


@dataclass
class ElectrodeInsertionMaker(electrode_flows.ElectrodeInsertionMaker):
    """Attempt ion insertion into a structure.

    The basic unit for cation insertion is:
        [get_stable_inserted_structure]:
            (static) -> N x (chgcar analysis -> relax) -> (return best structure)

    The workflow is:
        [relax structure]
        [get_stable_inserted_structure]
        [get_stable_inserted_structure]
        ... until the insertion is no longer topotactic.


    Parameters
    ----------
    name: str
        The name of the flow created by this maker.
    relax_maker: RelaxMaker
        A maker to perform relaxation calculations.
    static_maker: Maker
        A maker to perform static calculations.
    stucture_matcher: StructureMatcher
        The structure matcher to use to determine if additional insertion is needed.
    """

    @abstractmethod
    def get_charge_density(self, task_doc: TaskDoc) -> VolumetricData:
        """Get the charge density of a structure.

        Args:
            structure: The structure to get the charge density of.

        Returns
        -------
            The charge density.
        """
        aeccar0 = task_doc.calcs_reversed[0].output.aeccar0
        aeccar2 = task_doc.calcs_reversed[0].output.aeccar2
        return aeccar0 + aeccar2

    @abstractmethod
    def update_static_maker(self):
        """Ensure that the static maker will store the desired data."""
        self.static_maker.task_document_kwargs = (
            {"store_volumetric_data": ["aeccar"]},
        )
