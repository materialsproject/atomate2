"""Flows to generate EOS fits using CHGNet, M3GNet, or MACE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from monty.dev import deprecated

from atomate2.common.flows.eos import CommonEosMaker
from atomate2.forcefields import MLFF, _get_formatted_ff_name
from atomate2.forcefields.jobs import ForceFieldRelaxMaker

if TYPE_CHECKING:
    from jobflow import Maker
    from typing_extensions import Self


@dataclass
class ForceFieldEosMaker(CommonEosMaker):
    """
    Generate equation of state data using an ML forcefield.

    First relax a structure using relax_maker.
    Then perform a series of deformations on the relaxed structure, and
    relax atomic positions within the cell. For ML forcefields, there
    is no distinction between relax and static energies, unlike in a VASP
    calculation. Therefore these EosMakers default to static_maker = None.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    initial_relax_maker : .Maker | None
        Maker to relax the input structure, defaults to None (no initial relaxation).
    eos_relax_maker : .Maker
        Maker to relax deformed structures for the EOS fit.
    static_maker : .Maker | None
        Maker to generate statics after each relaxation, defaults to None.
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%.
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6.
    postprocessor : .job
        Optional postprocessing step, defaults to
        `atomate2.common.jobs.PostProcessEosEnergy`.
    _store_transformation_information : .bool = False
        Whether to store the information about transformations. Unfortunately
        needed at present to handle issues with emmet and pydantic validation
        TODO: remove this when clash is fixed
    """

    name: str = "Forcefield EOS Maker"
    initial_relax_maker: Maker = field(default_factory=ForceFieldRelaxMaker)
    eos_relax_maker: Maker = field(
        default_factory=lambda: ForceFieldRelaxMaker(relax_cell=False)
    )
    static_maker: Maker = None

    @classmethod
    def from_force_field_name(
        cls,
        force_field_name: str | MLFF,
        relax_initial_structure: bool = True,
    ) -> Self:
        """
        Create an EOS flow from a forcefield name.

        Parameters
        ----------
        force_field_name : str or .MLFF
            The name of the force field.
        relax_initial_structure: bool = True
            Whether to relax the initial structure before performing an EOS fit.

        Returns
        -------
        ForceFieldEosMaker
        """
        force_field_name = _get_formatted_ff_name(force_field_name)
        return cls(
            name=f"{force_field_name.split('MLFF.')[-1]} EOS Maker",
            initial_relax_maker=(
                ForceFieldRelaxMaker(force_field_name=force_field_name)
                if relax_initial_structure
                else None
            ),
            eos_relax_maker=ForceFieldRelaxMaker(
                force_field_name=force_field_name, relax_cell=False
            ),
            static_maker=None,
        )


@deprecated(
    replacement=ForceFieldEosMaker,
    deadline=(2025, 1, 1),
    message='Use ForceFieldEosMaker.from_force_field_name(force_field_name = "CHGNet")',
)
@dataclass
class CHGNetEosMaker(CommonEosMaker):
    """
    Generate equation of state data using the CHGNet ML forcefield.

    First relax a structure using relax_maker.
    Then perform a series of deformations on the relaxed structure, and
    relax atomic positions within the cell. For ML forcefields, there
    is no distinction between relax and static energies, unlike in a VASP
    calculation. Therefore these EosMakers default to static_maker = None.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    initial_relax_maker : .Maker | None
        Maker to relax the input structure, defaults to None (no initial relaxation).
    eos_relax_maker : .Maker
        Maker to relax deformed structures for the EOS fit.
    static_maker : .Maker | None
        Maker to generate statics after each relaxation, defaults to None.
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%.
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6.
    postprocessor : .job
        Optional postprocessing step, defaults to
        `atomate2.common.jobs.PostProcessEosEnergy`.
    _store_transformation_information : .bool = False
        Whether to store the information about transformations. Unfortunately
        needed at present to handle issues with emmet and pydantic validation
        TODO: remove this when clash is fixed
    """

    name: str = "CHGNet EOS Maker"
    initial_relax_maker: Maker = field(
        default_factory=lambda: ForceFieldRelaxMaker(force_field_name="CHGNet")
    )
    eos_relax_maker: Maker = field(
        default_factory=lambda: ForceFieldRelaxMaker(
            force_field_name="CHGNet", relax_cell=False
        )
    )
    static_maker: Maker = None


@deprecated(
    replacement=ForceFieldEosMaker,
    deadline=(2025, 1, 1),
    message='Use ForceFieldEosMaker.from_force_field_name(force_field_name = "M3GNet")',
)
@dataclass
class M3GNetEosMaker(CommonEosMaker):
    """
    Generate equation of state data using the M3GNet ML forcefield.

    First relax a structure using relax_maker.
    Then perform a series of deformations on the relaxed structure, and
    relax atomic positions within the cell. For ML forcefields, there
    is no distinction between relax and static energies, unlike in a VASP
    calculation. Therefore these EosMakers default to static_maker = None.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    initial_relax_maker : .Maker | None
        Maker to relax the input structure, defaults to None (no initial relaxation).
    eos_relax_maker : .Maker
        Maker to relax deformed structures for the EOS fit.
    static_maker : .Maker | None
        Maker to generate statics after each relaxation, defaults to None.
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%.
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6.
    postprocessor : .job
        Optional postprocessing step, defaults to
        `atomate2.common.jobs.PostProcessEosEnergy`.
    _store_transformation_information : .bool = False
        Whether to store the information about transformations. Unfortunately
        needed at present to handle issues with emmet and pydantic validation
        TODO: remove this when clash is fixed
    """

    name: str = "M3GNet EOS Maker"
    initial_relax_maker: Maker = field(
        default_factory=lambda: ForceFieldRelaxMaker(force_field_name="M3GNet")
    )
    eos_relax_maker: Maker = field(
        default_factory=lambda: ForceFieldRelaxMaker(
            force_field_name="M3GNet", relax_cell=False
        )
    )
    static_maker: Maker = None


@deprecated(
    replacement=ForceFieldEosMaker,
    deadline=(2025, 1, 1),
    message='Use ForceFieldEosMaker.from_force_field_name(force_field_name = "MACE")',
)
@dataclass
class MACEEosMaker(CommonEosMaker):
    """
    Generate equation of state data using the MACE ML forcefield.

    First relax a structure using relax_maker.
    Then perform a series of deformations on the relaxed structure, and
    relax atomic positions within the cell. For ML forcefields, there
    is no distinction between relax and static energies, unlike in a VASP
    calculation. Therefore these EosMakers default to static_maker = None.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    initial_relax_maker : .Maker | None
        Maker to relax the input structure, defaults to None (no initial relaxation).
    eos_relax_maker : .Maker
        Maker to relax deformed structures for the EOS fit.
    static_maker : .Maker | None
        Maker to generate statics after each relaxation, defaults to None.
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%.
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6.
    postprocessor : .job
        Optional postprocessing step, defaults to
        `atomate2.common.jobs.PostProcessEosEnergy`.
    _store_transformation_information : .bool = False
        Whether to store the information about transformations. Unfortunately
        needed at present to handle issues with emmet and pydantic validation
        TODO: remove this when clash is fixed
    """

    name: str = "MACE EOS Maker"
    initial_relax_maker: Maker = field(
        default_factory=lambda: ForceFieldRelaxMaker(force_field_name="MACE")
    )
    eos_relax_maker: Maker = field(
        default_factory=lambda: ForceFieldRelaxMaker(
            force_field_name="MACE", relax_cell=False
        )
    )
    static_maker: Maker = None
