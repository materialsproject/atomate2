"""
Module defining EOS workflows using legacy MP/atomate parameters.

MPGGA prefix = MP PBE-GGA compatible WFs

MPMetaGGA prefix = MP r2SCAN Meta-GGA compatible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.flows.EOS.base import EosMaker
from atomate2.vasp.jobs.EOS.mp import (
    MPGGADeformationMaker,
    MPGGAEosRelaxMaker,
    MPGGAEosStaticMaker,
    MPMetaGGADeformationMaker,
    MPMetaGGAEosPreRelaxMaker,
    MPMetaGGAEosRelaxMaker,
    MPMetaGGAEosStaticMaker,
)
from atomate2.vasp.sets.EOS import MPGGAEosRelaxSetGenerator

if TYPE_CHECKING:
    from jobflow import Maker

    from atomate2.vasp.jobs.base import BaseVaspMaker


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
                user_incar_settings={"ISIF": 3, "EDIFFG": -0.05}
            )
        )
    )
    relax_maker2: BaseVaspMaker = field(
        default_factory=lambda: MPGGAEosRelaxMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )


@dataclass
class MPGGAEosMaker(EosMaker):
    """
    Workflow to generate MP PBE-GGA compatible EOS flow.

    First relax a structure using relax_maker.
    Then perform a series of deformations to the relaxed structure, and
    evaluate single-point energies with static_maker.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation, defaults to DoubleRelaxMaker
    deformation_maker : .BaseVaspMaker
        Maker to generate deformations + single-points, defaults to TransmuterMaker
    static_maker : .BaseVaspMaker
        Optional Maker to generate statics from transmutation.
        Original atomate workflow did not include statics, including it here
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6
    """

    name: str = "MP GGA EOS Maker"
    relax_maker: Maker | None = field(default_factory=MPGGAEosDoubleRelaxMaker)
    deformation_maker: Maker = field(
        default_factory=lambda: MPGGADeformationMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    static_maker: Maker | None = field(
        default_factory=lambda: MPGGAEosStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6


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
    relax_maker2: BaseVaspMaker = field(
        default_factory=lambda: MPMetaGGAEosRelaxMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )


@dataclass
class MPMetaGGAEosMaker(EosMaker):
    """
    Workflow to generate atomate1, MP r2SCAN Meta-GGA EOS flow.

    First relax a structure using relax_maker.
    Then perform a series of deformations to the relaxed structure, and
    evaluate single-point energies with static_maker.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation, defaults to DoubleRelaxMaker
    deformation_maker : .BaseVaspMaker
        Maker to generate deformations + single-points, defaults to TransmuterMaker
    static_maker : .BaseVaspMaker
        Optional Maker to generate statics from transmutation.
        Original atomate workflow did not include statics, including it here
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6
    """

    name: str = "MP Meta-GGA EOS Maker"
    relax_maker: Maker | None = field(default_factory=MPMetaGGAEosDoubleRelaxMaker)
    deformation_maker: Maker = field(
        default_factory=lambda: MPMetaGGADeformationMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    static_maker: Maker | None = field(
        default_factory=lambda: MPMetaGGAEosStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6
