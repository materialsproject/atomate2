"""
Module defining equation of state workflows.

Modeled on the atomate bulk_modulus workflows.

Prefixes are also defined in atomate2.sets.eos and atomate2.jobs.eos:
- No prefix (EosDoubleRelaxMaker and EosMaker): atomate2 default parameters
- MPLegacy: legacy MP PBE-GGA-compatible parameters (very high k-point density)
- MPGGA: MP PBE-GGA compatible parameters
- MPMetaGGA: MP r2SCAN-meta-GGA compatible parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.common.flows.eos import CommonEosMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.eos import (
    EosRelaxMaker,
    MPGGAEosRelaxMaker,
    MPGGAEosStaticMaker,
    MPLegacyEosRelaxMaker,
    MPLegacyEosStaticMaker,
    MPMetaGGAEosPreRelaxMaker,
    MPMetaGGAEosRelaxMaker,
    MPMetaGGAEosStaticMaker,
)
from atomate2.vasp.sets.eos import (
    EosSetGenerator,
    MPGGAEosRelaxSetGenerator,
    MPLegacyEosRelaxSetGenerator,
    MPMetaGGAEosRelaxSetGenerator,
)

if TYPE_CHECKING:
    from jobflow import Maker

    from atomate2.vasp.jobs.base import BaseVaspMaker

# No prefix, atomate2 base parameters


@dataclass
class EosDoubleRelaxMaker(DoubleRelaxMaker):
    """
    Workflow to generate initial double relaxation for EOS fitting.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to use to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to use to generate the second relaxation.

    NB: WAVECARs are copied over to speed up calculation
    """

    name: str = "EOS double relax"
    relax_maker1: BaseVaspMaker | None = field(default_factory=EosRelaxMaker)
    relax_maker2: BaseVaspMaker = field(default_factory=EosRelaxMaker)


@dataclass
class EosMaker(CommonEosMaker):
    """
    Generate equation of state data with default atomate2 parameters.

    First relax a structure using relax_maker.
    Then perform a series of deformations on the relaxed structure, and
    evaluate single-point energies with static_maker.

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
    postprocessor : .atomate2.common.jobs.EOSPostProcessor
        Optional postprocessing step, defaults to
        `atomate2.common.jobs.PostProcessEosEnergy`.
    _store_transformation_information : .bool = False
        Whether to store the information about transformations. Unfortunately
        needed at present to handle issues with emmet and pydantic validation
        TODO: remove this when clash is fixed
    """

    name: str = "EOS Maker"
    initial_relax_maker: Maker = field(default_factory=EosDoubleRelaxMaker)
    eos_relax_maker: Maker | None = field(
        default_factory=lambda: EosRelaxMaker(
            input_set_generator=EosSetGenerator(
                user_incar_settings={"ISIF": 2},
            )
        )
    )


# MPLegacy prefix: legacy MP PBE-GGA


@dataclass
class MPLegacyEosDoubleRelaxMaker(DoubleRelaxMaker):
    """
    Workflow to generate initial MP legacy PBE-GGA double relaxation.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to use to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to use to generate the second relaxation.

    NB: WAVECARs are copied over to speed up calculation
    """

    name: str = "MP Legacy EOS double relax"
    relax_maker1: BaseVaspMaker | None = field(default_factory=MPLegacyEosRelaxMaker)
    relax_maker2: BaseVaspMaker = field(default_factory=MPLegacyEosRelaxMaker)


@dataclass
class MPLegacyEosMaker(CommonEosMaker):
    """
    Generate equation of state data with MP legacy PBE-GGA parameters.

    First relax a structure using relax_maker.
    Then perform a series of deformations on the relaxed structure, and
    evaluate single-point energies with static_maker.

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
    postprocessor : .atomate2.common.jobs.EOSPostProcessor
        Optional postprocessing step, defaults to
        `atomate2.common.jobs.PostProcessEosEnergy`.
    _store_transformation_information : .bool = False
        Whether to store the information about transformations. Unfortunately
        needed at present to handle issues with emmet and pydantic validation
        TODO: remove this when clash is fixed
    """

    name: str = "MP Legacy GGA EOS Maker"
    initial_relax_maker: Maker | None = field(
        default_factory=MPLegacyEosDoubleRelaxMaker
    )
    eos_relax_maker: Maker | None = field(
        default_factory=lambda: MPLegacyEosRelaxMaker(
            input_set_generator=MPLegacyEosRelaxSetGenerator(
                user_incar_settings={"ISIF": 2},
            )
        )
    )
    static_maker: Maker | None = field(default_factory=MPLegacyEosStaticMaker)


# MPGGA prefix: MP PBE-GGA compatible


@dataclass
class MPGGAEosDoubleRelaxMaker(DoubleRelaxMaker):
    """
    Workflow to generate initial MP PBE-GGA double relaxation for EOS fitting.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to use to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to use to generate the second relaxation.

    NB: WAVECARs are copied over to speed up calculation
    """

    name: str = "MP GGA EOS double relax"
    relax_maker1: BaseVaspMaker | None = field(
        default_factory=lambda: MPGGAEosRelaxMaker(
            input_set_generator=MPGGAEosRelaxSetGenerator(
                user_incar_settings={"EDIFFG": -0.05}
            )
        )
    )
    relax_maker2: BaseVaspMaker = field(default_factory=MPGGAEosRelaxMaker)


@dataclass
class MPGGAEosMaker(CommonEosMaker):
    """
    Generate equation of state data with MP PBE-GGA parameters.

    First relax a structure using relax_maker.
    Then perform a series of deformations on the relaxed structure, and
    evaluate single-point energies with static_maker.

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
    postprocessor : .atomate2.common.jobs.EOSPostProcessor
        Optional postprocessing step, defaults to
        `atomate2.common.jobs.PostProcessEosEnergy`.
    _store_transformation_information : .bool = False
        Whether to store the information about transformations. Unfortunately
        needed at present to handle issues with emmet and pydantic validation
        TODO: remove this when clash is fixed
    """

    name: str = "MP GGA EOS Maker"
    initial_relax_maker: Maker | None = field(default_factory=MPGGAEosDoubleRelaxMaker)
    eos_relax_maker: Maker | None = field(
        default_factory=lambda: MPGGAEosRelaxMaker(
            input_set_generator=MPGGAEosRelaxSetGenerator(
                user_incar_settings={"ISIF": 2}
            )
        )
    )
    static_maker: Maker | None = field(default_factory=MPGGAEosStaticMaker)


@dataclass
class MPMetaGGAEosDoubleRelaxMaker(DoubleRelaxMaker):
    """
    Workflow to generate initial MP r2SCAN Meta-GGA double relaxation for EOS fitting.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to use to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to use to generate the second relaxation.

    NB: WAVECARs are copied over to speed up calculation
    """

    name: str = "MP Meta-GGA EOS double relax"
    relax_maker1: BaseVaspMaker | None = field(
        default_factory=MPMetaGGAEosPreRelaxMaker
    )
    relax_maker2: BaseVaspMaker = field(default_factory=MPMetaGGAEosRelaxMaker)


@dataclass
class MPMetaGGAEosMaker(CommonEosMaker):
    """
    Generate equation of state data with MP r2SCAN-meta-GGA parameters.

    First relax a structure using relax_maker.
    Then perform a series of deformations on the relaxed structure, and
    evaluate single-point energies with static_maker.

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
    postprocessor : .atomate2.common.jobs.EOSPostProcessor
        Optional postprocessing step, defaults to
        `atomate2.common.jobs.PostProcessEosEnergy`.
    _store_transformation_information : .bool = False
        Whether to store the information about transformations. Unfortunately
        needed at present to handle issues with emmet and pydantic validation
        TODO: remove this when clash is fixed
    """

    name: str = "MP Meta-GGA EOS Maker"
    initial_relax_maker: Maker | None = field(
        default_factory=MPMetaGGAEosDoubleRelaxMaker
    )
    eos_relax_maker: Maker | None = field(
        default_factory=lambda: MPMetaGGAEosRelaxMaker(
            input_set_generator=MPMetaGGAEosRelaxSetGenerator(
                user_incar_settings={"ISIF": 2}
            )
        )
    )
    static_maker: Maker | None = field(default_factory=MPMetaGGAEosStaticMaker)
