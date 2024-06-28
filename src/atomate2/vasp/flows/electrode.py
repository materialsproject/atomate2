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
        [get_stable_inserted_structure]
        ... until the insertion is no longer topotactic.

    If you use this workflow please cite the following paper:
        Shen, J.-X., Horton, M., & Persson, K. A. (2020).
        A charge-density-based general cation insertion algorithm for
        generating new Li-ion cathode materials.
        npj Computational Materials, 6(161), 1—7.
        doi: 10.1038/s41524-020-00422-3


    Attributes
    ----------
    name: str
        The name of the flow created by this maker.
    relax_maker: RelaxMaker
        A maker to perform relaxation calculations.
    bulk_relax_maker: Maker
        A separate maker to perform the first bulk relaxation calculation.
        If None, the relax_maker will be used.
    static_maker: Maker
        A maker to perform static calculations.
    structure_matcher: StructureMatcher
        The structure matcher to use to determine if additional insertion is needed.
    """

    def get_charge_density(self, prev_dir: Path | str) -> VolumetricData:
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
        self.static_maker.task_document_kwargs["store_volumetric_data"] = (
            store_volumetric_data
        )
