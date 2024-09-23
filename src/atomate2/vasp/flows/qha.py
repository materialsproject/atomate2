"""Define QHA flow for VASP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from atomate2.common.flows.qha import CommonQhaMaker
from atomate2.vasp.flows.phonons import PhononMaker
from atomate2.vasp.jobs.core import TightRelaxMaker
from atomate2.vasp.jobs.eos import EosRelaxMaker
from atomate2.vasp.sets.core import TightRelaxSetGenerator


@dataclass
class QhaMaker(CommonQhaMaker):
    """
    Perform quasi-harmonic approximation.

    First relax a structure using relax_maker.
    Then perform a series of deformations on the relaxed structure, and
    then compute harmonic phonons for each deformed structure.
    Finally, compute Gibb's free energy.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    initial_relax_maker: .TightRelaxMaker | None
        Maker to relax the input structure.
    eos_relax_maker: .TightRelaxMaker | None
        Maker to relax deformed structures for the EOS fit.
        The volume has to be fixed!
    phonon_maker: .PhononMaker
        Maker to compute phonons. The volume has to be fixed!
        The beforehand relaxation could be switched off.
    linear_strain: tuple[float, float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%.
    number_of_frames: int
        Number of strain calculations to do for EOS fit, default = 6.
    t_max: float | None
        Maximum temperature until which the QHA will be performed
    pressure: float | None
        Pressure at which the QHA will be performed (default None, no pressure)
    skip_analysis: bool
        Skips the analysis step and only performs EOS and phonon computations.
    ignore_imaginary_modes: bool
        By default, volumes where the harmonic phonon approximation shows imaginary
        will be ignored
    eos_type: supported_eos
        Equation of State type used for the fitting. Defaults to vinet.


    """

    name: str = "VASP QHA Maker"
    initial_relax_maker: TightRelaxMaker | None = field(default_factory=TightRelaxMaker)
    eos_relax_maker: TightRelaxMaker | None = field(
        default_factory=lambda: EosRelaxMaker(
            input_set_generator=TightRelaxSetGenerator(
                user_incar_settings={"ISIF": 2},
            )
        )
    )
    phonon_maker: PhononMaker | None = field(
        default_factory=lambda: PhononMaker(bulk_relax_maker=None)
    )
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6
    pressure: float | None = None
    t_max: float | None = None
    ignore_imaginary_modes: bool = False
    skip_analysis: bool = False
    eos_type: Literal["vinet", "birch_murnaghan", "murnaghan"] = "vinet"
    analyze_free_energy_kwargs: dict = field(default_factory=dict)

    @property
    def prev_calc_dir_argname(self) -> str:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return "prev_dir"
