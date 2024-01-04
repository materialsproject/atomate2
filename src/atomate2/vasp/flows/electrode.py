"""Flow for electrode analysis with specific VASP implementations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pymatgen.io.vasp.outputs import Chgcar

from atomate2.common.flows import electrode as electrode_flows
from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
    from pymatgen.io.vasp.outputs import VolumetricData

logger = logging.getLogger(__name__)


class ElectrodeInsertionMaker(electrode_flows.ElectrodeInsertionMaker):
    """Attempt ion insertion into a structure.

    The basic unit for cation insertion is:
        [get_stable_inserted_structure]:
            (static) -> (chgcar analysis) ->
            N x (relax) -> (return best structure)

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

    def get_charge_density(self, prev_dir) -> VolumetricData:
        """Get the charge density of a structure.

        Parameters
        ----------
        prev_dir:
            The previous directory where the static calculation was performed.

        Returns
        -------
            The charge density.
        """
        prev_dir = Path(strip_hostname(prev_dir))
        aeccar0 = Chgcar.from_file(prev_dir / "AECCAR0.gz")
        aeccar2 = Chgcar.from_file(prev_dir / "AECCAR2.gz")
        return aeccar0 + aeccar2

    def update_static_maker(self) -> None:
        """Ensure that the static maker will store the desired data."""
        store_volumetric_data = list(
            self.static_maker.task_document_kwargs.get("store_volumetric_data", [])
        )
        store_volumetric_data.extend(["aeccar0", "aeccar2"])
        self.static_maker.task_document_kwargs[
            "store_volumetric_data"
        ] = store_volumetric_data
