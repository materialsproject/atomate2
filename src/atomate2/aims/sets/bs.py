"""Input sets for band structure calculations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pymatgen.core import Structure

from atomate2.aims.sets.base import AimsInputGenerator
from atomate2.aims.utils.bands import prepare_band_input

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Molecule


@dataclass
class BandStructureSetGenerator(AimsInputGenerator):
    """A generator for the band structure calculation input set.

    Parameters
    ----------
    calc_type: str
        The type of calculations
    k_point_density: float
        The number of k_points per angstrom
    """

    calc_type: str = "bands"
    k_point_density: float = 20

    def get_parameter_updates(
        self, structure: Structure, prev_parameters: dict[str, Any]
    ) -> dict[str, Sequence[str]]:
        """Get the parameter updates for the calculation.

        Parameters
        ----------
        structure: Structure
            The structure to calculate the bands for
        prev_parameters: Dict[str, Any]
            The previous parameters

        Returns
        -------
        The updated for the parameters for the output section of FHI-aims
        """
        updated_outputs = prev_parameters.get("output", [])
        updated_outputs += prepare_band_input(structure, self.k_point_density)
        return {"output": updated_outputs}


@dataclass
class GWSetGenerator(AimsInputGenerator):
    """
    A generator for the input set for calculations employing GW self-energy correction.

    Parameters
    ----------
    calc_type: str
        The type of calculations
    k_point_density: float
        The number of k_points per angstrom
    """

    calc_type: str = "GW"
    k_point_density: float = 20

    def get_parameter_updates(
        self, structure: Structure | Molecule, prev_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Get the parameter updates for the calculation.

        Parameters
        ----------
        atoms: Structure or Molecule
            The structure to calculate the bands for
        prev_parameters: Dict[str, Any]
            The previous parameters

        Returns
        -------
        The updated for the parameters for the output section of FHI-aims
        """
        updates = {"anacon_type": "two-pole"}
        current_output = prev_parameters.get("output", [])
        if isinstance(structure, Structure) and all(structure.lattice.pbc):
            updates.update(
                {
                    "qpe_calc": "gw_expt",
                    "output": current_output
                    + prepare_band_input(structure, self.k_point_density),
                }
            )
        else:
            updates.update(
                {
                    "qpe_calc": "gw",
                }
            )
        return updates
