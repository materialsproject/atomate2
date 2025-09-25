"""Core DFPT abinit flow makers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import abipy.core.abinit_units as abu
from abipy.abio.factories import scf_for_phonons

from atomate2.abinit.flows.dfpt_base import DfptFlowMaker
from atomate2.abinit.jobs.anaddb import AnaddbDfptDteMaker, AnaddbPhBandsDOSMaker
from atomate2.abinit.jobs.core import StaticMaker
from atomate2.abinit.jobs.mrgdv import MrgdvMaker
from atomate2.abinit.jobs.response import (
    DdeMaker,
    DdkMaker,
    PhononResponseMaker,
    WfqMaker,
)
from atomate2.abinit.sets.core import ShgStaticSetGenerator, StaticSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Flow, Maker
    from pymatgen.core.structure import Structure

    from atomate2.abinit.jobs.base import BaseAbinitMaker


@dataclass
class ShgFlowMaker(DfptFlowMaker):
    """
    Maker to compute the static DFPT second-harmonic generation tensor.

    Maker to compute the electronic contribution to the
        static DFPT second-harmonic generation tensor.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    scissor: float
        A rigid shift of the conduction bands in eV.
    """

    name: str = "DFPT Chi2 SHG"
    anaddb_maker: Maker | None = field(default_factory=AnaddbDfptDteMaker)
    use_dde_sym: bool = False
    static_maker: BaseAbinitMaker = field(
        default_factory=lambda: StaticMaker(input_set_generator=ShgStaticSetGenerator())
    )

    def make(
        self,
        structure: Structure | None = None,
        restart_from: str | Path | None = None,
    ) -> Flow:
        """
        Create a DFPT flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        restart_from : str or Path or None
            One previous directory to restart from.

        Returns
        -------
        Flow
            A DFPT flow
        """
        return super().make(structure=structure, restart_from=restart_from)

    @classmethod
    def with_scissor(cls, scissor: float) -> ShgFlowMaker:
        """
        Create a DFPT Flow to compute the static SHG tensor with a scissor correction.

        Create a DFPT Flow to compute the electronic contribution
            to the static SHG tensor with a scissor correction.

        Parameters
        ----------
        scissor : float
            The scissor-correction to the band gap in eV.

        Returns
        -------
        Flow
            An ShgFlowMaker Flow
        """
        return cls(
            static_maker=StaticMaker(
                input_set_generator=ShgStaticSetGenerator(
                    user_abinit_settings={
                        "nstep": 500,
                        "toldfe": 1e-22,
                        "autoparal": 1,
                        "npfft": 1,
                        "dfpt_sciss": scissor * abu.eV_Ha,
                    },
                    factory_kwargs={
                        "smearing": "nosmearing",
                        "spin_mode": "unpolarized",
                        "kppa": 3000,
                        "nbdbuf": 0,
                    },
                )
            ),
            name="DFPT Chi2 SHG with scissor",
        )


@dataclass
class PhononMaker(DfptFlowMaker):
    """
    Maker to generate a phonon band structure and phonon DOS flow with abinit.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    with_dde : bool
        True if the DDE calculations should be included, False otherwise.
    run_anaddb : bool
        True if the anaddb calculations should be included, False otherwise.
    run_mrgddb : bool
        True if the merge of DDB files should be included, False otherwise.
    run_mrgdv : bool
        True if the merge of POT files should be included, False otherwise.
    """

    name: str = "Phonon Flow"
    with_dde: bool = True
    run_anaddb: bool = True
    run_mrgdv: bool = False
    static_maker: BaseAbinitMaker = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=StaticSetGenerator(factory=scf_for_phonons)
        )
    )
    ddk_maker: BaseAbinitMaker = field(default_factory=DdkMaker)
    dde_maker: BaseAbinitMaker = field(default_factory=DdeMaker)
    phonon_maker: BaseAbinitMaker = field(default_factory=PhononResponseMaker)
    mrgdv_maker: Maker | None = field(default_factory=MrgdvMaker)
    anaddb_maker: Maker | None = field(default_factory=AnaddbPhBandsDOSMaker)
    wfq_maker: BaseAbinitMaker = field(default_factory=WfqMaker)
    qptopt: int | None = 1

    def __post_init__(self) -> None:
        """Process post-init configuration."""
        if not self.with_dde:
            self.ddk_maker = None
            self.dde_maker = None

        if not self.run_mrgdv:
            self.mrgdv_maker = None

        if not self.run_anaddb:
            self.anaddb_maker = None
