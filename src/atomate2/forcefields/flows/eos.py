"""Flows to generate EOS fits using CHGNet, M3GNet, or MACE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.common.flows.eos import CommonEosMaker
from atomate2.forcefields.jobs import CHGNetRelaxMaker, M3GNetRelaxMaker, MACERelaxMaker

if TYPE_CHECKING:
    from jobflow import Maker


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
        Maker to relax deformationed structures for the EOS fit.
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
    initial_relax_maker: Maker = field(default_factory=CHGNetRelaxMaker)
    eos_relax_maker: Maker = field(
        default_factory=lambda: CHGNetRelaxMaker(relax_cell=False)
    )
    static_maker: Maker = None


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
        Maker to relax deformationed structures for the EOS fit.
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
    initial_relax_maker: Maker = field(default_factory=M3GNetRelaxMaker)
    eos_relax_maker: Maker = field(
        default_factory=lambda: M3GNetRelaxMaker(relax_cell=False)
    )
    static_maker: Maker = None


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
        Maker to relax deformationed structures for the EOS fit.
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
    initial_relax_maker: Maker = field(default_factory=MACERelaxMaker)
    eos_relax_maker: Maker = field(
        default_factory=lambda: MACERelaxMaker(relax_cell=False)
    )
    static_maker: Maker = None
