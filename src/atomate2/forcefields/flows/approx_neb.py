"""Run an ApproxNEB flow using MLIPs."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pymatgen.io.common import VolumetricData
from pymatgen.io.vasp.outputs import Chgcar
from typing_extensions import Self

from atomate2.common.flows.approx_neb import ApproxNebFromEndpointsMaker
from atomate2.forcefields import MLFF, _get_formatted_ff_name
from atomate2.forcefields.jobs import ForceFieldRelaxMaker


@dataclass
class MLFFApproxNebFromEndpointsMaker(ApproxNebFromEndpointsMaker):
    """
    Perform ApproxNEB on a single hop using ML forcefields.

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
    """

    image_relax_maker: ForceFieldRelaxMaker
    name: str = "MLFF ApproxNEB single hop from endpoints maker"

    def get_charge_density(
        self, prev_dir_or_chgcar: str | Path | Chgcar | Sequence[str | Path | Chgcar]
    ) -> VolumetricData:
        """Obtain charge density from a specified path, CHGCAR, or list of them.

        Parameters
        ----------
        prev_dir_or_chgcar : str or Path or pymatgen .Chgcar, or a list of these
            Path(s) to the CHGCAR/AECCAR* file(s) or the object(s) themselves.

        Returns
        -------
            VolumetricData
                The charge density
        """
        if isinstance(prev_dir_or_chgcar, str | Path | Chgcar):
            prev_dir_or_chgcar = [prev_dir_or_chgcar]

        for idx, obj in enumerate(prev_dir_or_chgcar):
            chg = Chgcar.from_file(obj) if isinstance(obj, str | Path) else obj
            if idx == 0:
                charge_density = chg
            else:
                charge_density += chg
        return charge_density

    @classmethod
    def from_force_field_name(
        cls,
        force_field_name: str | MLFF,
        **kwargs,
    ) -> Self:
        """
        Create an ApproxNEB flow from a forcefield name.

        Parameters
        ----------
        force_field_name : str or .MLFF
            The name of the force field.
        **kwargs
            Additional kwargs to pass to ApproxNEB


        Returns
        -------
        MLFFApproxNebFromEndpointsMaker
        """
        force_field_name = _get_formatted_ff_name(force_field_name)
        kwargs.update(
            image_relax_maker=ForceFieldRelaxMaker(
                force_field_name=force_field_name, relax_cell=False
            ),
        )
        return cls(
            name=(
                f"{force_field_name.split('MLFF.')[-1]} ApproxNEB from endpoints Maker"
            ),
            **kwargs,
        )
