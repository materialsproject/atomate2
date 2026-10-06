"""Define the force field CALPHAD maker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ase import units
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from emmet.core.types.enums import StoreTrajectoryOption

from atomate2.ase.md import MDEnsemble
from atomate2.common.flows.calphad import BaseCalphadMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from atomate2.forcefields.md import ForceFieldMDMaker

if TYPE_CHECKING:
    from typing_extensions import Self

    from atomate2.forcefields import MLFF

_DEFAULT_FORCE_FIELD = "MACE-MP-0"


def _get_liquid_md_maker(
    calculator: dict[str, Any], temperature: float, n_steps: int, **kwargs
) -> ForceFieldMDMaker:
    """Get an isotropic NPT maker at zero pressure with a 2 fs time step."""
    return ForceFieldMDMaker(
        store_trajectory=StoreTrajectoryOption.NO,
        zero_linear_momentum=True,
        ensemble=MDEnsemble.npt,
        # AseMDMaker takes the class, its type hint says an instance
        dynamics=IsotropicMTKNPT,  # type: ignore[arg-type]
        temperature=temperature,
        pressure=0.0,
        time_step=2.0,
        n_steps=n_steps,
        ase_md_kwargs={"tdamp": 200 * units.fs, "pdamp": 2000 * units.fs},
        **calculator,
        **kwargs,
    )


def _get_makers(
    force_field_name: str | MLFF | dict,
    melt_temperature: float | None = None,
    liquid_temperature: float | None = None,
    calculator_kwargs: dict | None = None,
) -> dict:
    """Get the relaxation and liquid MD makers for one force field."""
    calculator: dict[str, Any] = {
        "force_field_name": force_field_name,
        "calculator_kwargs": dict(calculator_kwargs or {}),
    }
    makers: dict[str, Any] = {
        "relax_maker": ForceFieldRelaxMaker(
            relax_cell=True, steps=2000, relax_kwargs={"fmax": 0.001}, **calculator
        )
    }
    if melt_temperature is not None and liquid_temperature is not None:
        makers["liquid_melt_maker"] = _get_liquid_md_maker(
            calculator, melt_temperature, 5000, name="liquid melt"
        )
        makers["liquid_md_maker"] = _get_liquid_md_maker(
            calculator,
            liquid_temperature,
            10000,
            name="liquid MD",
            ionic_step_data=("energy", "structure"),
            traj_interval=10,
        )
    return makers


@dataclass
class CalphadMaker(BaseCalphadMaker):
    """
    Maker to fit a CALPHAD database for a binary system with a force field.

    The solid SQS are relaxed, including the cell, with fmax = 0.001 eV/A for
    at most 2000 steps. Each liquid SQS is doubled along each lattice
    vector. It is melted for 10 ps at melt_temperature and then run for 20 ps at
    liquid_temperature. Both runs are isotropic NPT at zero pressure with a 2 fs
    time step and no net momentum. The first 5 ps of the second run are left out
    of the mean energy.

    By default, all steps use MACE-MP-0. Use :obj:`from_force_field_name` to
    run every step with another force field. The liquid makers are only set
    by :obj:`from_force_field_name`, since their temperatures depend on the
    system.

    See :obj:`.BaseCalphadMaker` for what the TDB file contains.

    This workflow is new and has not been tested widely. It might still change
    in future versions.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    lattices : list[str]
        ATAT lattice names.
    level : int
        Level of the ATAT SQS database.
    terms : dict[str, list[str]]
        Lines of the sqs2tdb terms.in file for each lattice.
    relax_maker : .ForceFieldRelaxMaker
        Maker to relax the solid SQS.
    liquid_melt_maker : .ForceFieldMDMaker or None
        MD maker that melts the liquid SQS.
    liquid_md_maker : .ForceFieldMDMaker or None
        MD maker for the liquid energy. It must store the energy and structure
        of each frame.
    n_equilibration_frames : int
        Number of stored frames at the start of the liquid MD that are left out
        of the mean energy.
    liquid_supercell : int
        Number of repeats of each liquid SQS along each lattice vector.
    """

    relax_maker: ForceFieldRelaxMaker = field(
        default_factory=lambda: ForceFieldRelaxMaker(
            force_field_name=_DEFAULT_FORCE_FIELD,
            relax_cell=True,
            steps=2000,
            relax_kwargs={"fmax": 0.001},
        )
    )
    n_equilibration_frames: int = 250
    liquid_supercell: int = 2

    @classmethod
    def from_force_field_name(
        cls,
        force_field_name: str | MLFF | dict,
        melt_temperature: float | None = None,
        liquid_temperature: float | None = None,
        calculator_kwargs: dict | None = None,
        **kwargs,
    ) -> Self:
        """
        Create a CALPHAD maker that uses one force field for all steps.

        Parameters
        ----------
        force_field_name : str or .MLFF or dict
            The name of the force field.
        melt_temperature : float or None
            Temperature in K at which the liquid SQS are melted.
        liquid_temperature : float or None
            Temperature in K of the liquid MD used for the energy.
        calculator_kwargs : dict or None
            Keyword arguments passed to the force field calculator.
        **kwargs
            Further keyword arguments passed to CalphadMaker. A maker given here
            replaces the one built for the force field.

        Returns
        -------
        CalphadMaker
        """
        makers = _get_makers(
            force_field_name, melt_temperature, liquid_temperature, calculator_kwargs
        )
        return cls(**{**makers, **kwargs})
