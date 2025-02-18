"""Define the ApproxNEB VASP flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.common.flows.approx_neb import (
    ApproxNebFromEndpointsMaker,
    CommonApproxNebMaker,
)
from atomate2.vasp.jobs.approx_neb import (
    ApproxNebHostRelaxMaker,
    ApproxNebImageRelaxMaker,
    get_charge_density,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.io.vasp.outputs import Chgcar

    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class ApproxNebMaker(CommonApproxNebMaker):
    """Run an ApproxNEB workflow with VASP.

    Parameters
    ----------
    name : str = "ApproxNEB"
        Name of the workflow
    host_relax_maker : Maker
        Optional, a maker to relax the input host structure.
        Defaults to atomate2.vasp.jobs.approx_neb.ApproxNebHostRelaxMaker
    image_relax_maker : maker
        Required, a maker to relax the ApproxNEB endpoints and images.
        Defaults to atomate2.vasp.jobs.approx_neb.ApproxNebImageRelaxMaker
    selective_dynamics_scheme : "fix_two_atoms" (default) or None
        If "fix_two_atoms", uses the default selective dynamics scheme of ApproxNEB,
        wherein the migrating ion and the ion farthest from it are the only
        ions whose positions can relax.
    use_aeccar : bool = False
        If True, the sum of the host structure AECCAR0 (pseudo-core charge density)
        and AECCAR2 (valence charge density) are used in image pathfinding.
        If False (default), the CHGCAR (valence charge density) is used.
    min_hop_distance : float or bool (default = True)
        If a float, skips any hops where the working ion moves a distance less
        than min_hop_distance.
        If True, min_hop_distance is set to twice the average ionic radius.
        If False, no checks are made.
    """

    name: str = "ApproxNEB VASP"
    host_relax_maker: BaseVaspMaker | None = field(
        default_factory=ApproxNebHostRelaxMaker
    )
    image_relax_maker: BaseVaspMaker = field(default_factory=ApproxNebImageRelaxMaker)
    use_aeccar: bool = False

    def get_charge_density(self, prev_dir: str | Path) -> Chgcar:
        """Get charge density from a prior VASP calculation.

        Parameters
        ----------
        prev_dir : str or Path
            Path to the previous VASP calculation

        Returns
        -------
        pymatgen Chgcar object
        """
        return get_charge_density(prev_dir, use_aeccar=self.use_aeccar)


@dataclass
class ApproxNebSingleHopMaker(ApproxNebFromEndpointsMaker):
    """
    Create an ApproxNEB VASP flow from specified endpoints.

    image_relax_maker : Maker
        Maker to relax both endpoints and images
    selective_dynamics_scheme : "fix_two_atoms" (default) or None
        If "fix_two_atoms", uses the default selective dynamics scheme of ApproxNEB,
        wherein the migrating ion and the ion farthest from it are the only
        ions whose positions can relax.
    min_images_per_hop : int or None
        If an int, the minimum number of image calculations per hop that
        must succeed to mark a hop as successfully calculated.
    min_hop_distance : float or bool (default = True)
        If a float, skips any hops where the working ion moves a distance less
        than min_hop_distance.
        If True, min_hop_distance is set to twice the average ionic radius.
        If False, no checks are made.
    use_aeccar : bool = False
        If True, the sum of the host structure AECCAR0 (pseudo-core charge density)
        and AECCAR2 (valence charge density) are used in image pathfinding.
        If False (default), the CHGCAR (valence charge density) is used.
    """

    image_relax_maker: BaseVaspMaker = field(default_factory=ApproxNebImageRelaxMaker)
    name: str = "VASP ApproxNEB single hop from endpoints maker"
    use_aeccar: bool = False

    def get_charge_density(self, prev_dir: str | Path) -> Chgcar:
        """Get charge density from a prior VASP calculation.

        Parameters
        ----------
        prev_dir : str or Path
            Path to the previous VASP calculation

        Returns
        -------
        pymatgen Chgcar object
        """
        return get_charge_density(prev_dir, use_aeccar=self.use_aeccar)
