"""Core DFPT abinit flow makers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import abipy.core.abinit_units as abu

from atomate2.abinit.flows.dfpt_base import DfptFlowMaker
from atomate2.abinit.jobs.anaddb import AnaddbDfptDteMaker, AnaddbPhBandsDOSMaker
from atomate2.abinit.jobs.core import StaticMaker, WfqMaker
from atomate2.abinit.jobs.mrgdv import MrgdvMaker
from atomate2.abinit.jobs.response import (
    DdeMaker,
    DdkMaker,
    DteMaker,
    PhononResponseMaker,
)
from atomate2.abinit.sets.core import PhononsStaticSetGenerator, ShgStaticSetGenerator

if TYPE_CHECKING:
    from jobflow import Maker

    from atomate2.abinit.jobs.base import BaseAbinitMaker

__all__ = ["PhononMaker", "ShgFlowMaker"]


@dataclass
class ShgFlowMaker(DfptFlowMaker):
    """
    Maker to compute the static second-harmonic generation (SHG) tensor.

    This maker uses DFPT to compute the electronic contribution to the static
    second-harmonic generation (chi-2) tensor. The workflow includes DDK, DDE,
    and DTE calculations, followed by ANADDB analysis to extract the SHG tensor.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    use_dde_sym : bool
        If True, use symmetries for DDE calculations. Default is False
        (required for DTE calculations).
    static_maker : .BaseAbinitMaker
        The maker for the static calculation. Defaults to StaticMaker with
        ShgStaticSetGenerator.
    ddk_maker : .BaseAbinitMaker
        The maker for DDK calculations. Defaults to DdkMaker.
    dde_maker : .BaseAbinitMaker
        The maker for DDE calculations. Defaults to DdeMaker.
    dte_maker : .BaseAbinitMaker
        The maker for DTE calculations. Defaults to DteMaker.
    anaddb_maker : .Maker or None
        The maker to analyze the merged DDB with ANADDB. Defaults to
        AnaddbDfptDteMaker.
    """

    name: str = "DFPT Chi2 SHG"
    use_dde_sym: bool = False
    static_maker: BaseAbinitMaker = field(
        default_factory=lambda: StaticMaker(input_set_generator=ShgStaticSetGenerator())
    )
    ddk_maker: BaseAbinitMaker = field(default_factory=DdkMaker)
    dde_maker: BaseAbinitMaker = field(default_factory=DdeMaker)
    dte_maker: BaseAbinitMaker = field(default_factory=DteMaker)
    anaddb_maker: Maker | None = field(default_factory=AnaddbDfptDteMaker)

    @classmethod
    def with_scissor(cls, scissor: float) -> ShgFlowMaker:
        """
        Create an ShgFlowMaker with scissor correction to the band gap.

        This classmethod creates a maker configured to compute the electronic
        contribution to the static SHG tensor with a rigid shift applied to
        the conduction bands (scissor operator). The scissor correction is
        useful when the ground-state calculation underestimates the band gap.

        Parameters
        ----------
        scissor : float
            The scissor correction to apply to the band gap, in eV. This
            value rigidly shifts the conduction bands upward to correct for
            band gap underestimation.

        Returns
        -------
        ShgFlowMaker
            An ShgFlowMaker instance configured with the specified scissor
            correction and optimized calculation parameters.
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
    Maker to generate phonon band structure and density of states flow.

    This maker uses DFPT to compute phonon properties including band structure
    and DOS. It can optionally include DDE calculations (for Born effective
    charges and dielectric tensor), WFQ calculations, and ANADDB post-processing
    for interpolated phonon bands and thermodynamic properties.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    with_dde : bool
        If True, include DDK and DDE calculations to compute Born effective
        charges and dielectric tensor. Default is True.
    run_anaddb : bool
        If True, run ANADDB to analyze the merged DDB and compute phonon
        bands, DOS, and thermodynamic properties. Default is True.
    run_mrgdv : bool
        If True, merge first-order potential (POT) files. Default is False.
    static_maker : .BaseAbinitMaker
        The maker for the static calculation. Defaults to StaticMaker with
        PhononsStaticSetGenerator.
    ddk_maker : .BaseAbinitMaker or None
        The maker for DDK calculations. Defaults to DdkMaker. Set to None
        if with_dde is False.
    dde_maker : .BaseAbinitMaker or None
        The maker for DDE calculations. Defaults to DdeMaker. Set to None
        if with_dde is False.
    phonon_maker : .BaseAbinitMaker
        The maker for phonon (atomic displacement) calculations. Defaults
        to PhononResponseMaker.
    mrgdv_maker : .Maker or None
        The maker to merge POT files. Defaults to MrgdvMaker.
    anaddb_maker : .Maker or None
        The maker to analyze the merged DDB with ANADDB. Defaults to
        AnaddbPhBandsDOSMaker.
    wfq_maker : .BaseAbinitMaker
        The maker for wavefunctions at q-points calculations. Defaults to
        WfqMaker.
    qptopt : int or None
        Option for q-point generation. Default is 1.

    Notes
    -----
    Q-point mesh specification (inherited from DfptFlowMaker):
        You can specify the phonon q-point mesh using one of four methods
        (only one can be used at a time):

        - Default (recommended for high-throughput):
            If qpt_list, ngqpt, and user_qpoints_settings are all None,
            the q-point grid automatically matches the k-point mesh from
            the ground state calculation. This ensures consistency and
            enables full anaddb post-processing.
        - qpt_list : list[list]
            Explicit list of q-points for phonon calculations.
            Note: anaddb post-processing (phonon bands and DOS) is not
            available with explicit q-point lists. For full phonon
            analysis, use ngqpt or user_qpoints_settings to define a
            uniform q-point grid.
        - ngqpt : list
            Monkhorst-Pack grid divisions (e.g., [4, 4, 4]) to explicitly
            define a uniform q-point grid different from the k-point mesh.
        - user_qpoints_settings : dict or KSampling
            Custom q-point settings for uniform grids, e.g.,
            {"reciprocal_density": 1000} or a KSampling object.
    """

    name: str = "Phonon Flow"
    with_dde: bool = True
    run_anaddb: bool = True
    run_mrgdv: bool = False
    static_maker: BaseAbinitMaker = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=PhononsStaticSetGenerator()
        )
    )
    ddk_maker: BaseAbinitMaker | None = field(default_factory=DdkMaker)
    dde_maker: BaseAbinitMaker | None = field(default_factory=DdeMaker)
    phonon_maker: BaseAbinitMaker = field(default_factory=PhononResponseMaker)
    mrgdv_maker: Maker | None = field(default_factory=MrgdvMaker)
    anaddb_maker: Maker | None = field(default_factory=AnaddbPhBandsDOSMaker)
    wfq_maker: BaseAbinitMaker = field(default_factory=WfqMaker)
    qptopt: int | None = 1

    def __post_init__(self) -> None:
        """
        Configure makers based on run flags.

        This method conditionally disables certain calculations based on the
        with_dde, run_mrgdv, and run_anaddb flags. When with_dde is False,
        both DDK and DDE makers are set to None. When run_mrgdv is False,
        the mrgdv_maker is set to None. When run_anaddb is False, the
        anaddb_maker is set to None.
        """
        if not self.with_dde:
            # DDK is a prerequisite for DDE, so both are disabled together
            self.ddk_maker = None
            self.dde_maker = None

        if not self.run_mrgdv:
            self.mrgdv_maker = None

        if not self.run_anaddb:
            self.anaddb_maker = None
