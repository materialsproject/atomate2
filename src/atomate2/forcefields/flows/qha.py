"""Define QHA flow for forcefields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from atomate2.common.flows.qha import CommonQhaMaker
from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker

if TYPE_CHECKING:
    from typing_extensions import Self

    from atomate2.forcefields import MLFF


@dataclass
class ForceFieldQhaMaker(CommonQhaMaker):
    """
    Perform quasi-harmonic approximation with a machine learning forcefield.

    First relax a structure using relax_maker.
    Then perform a series of deformations on the relaxed structure, and
    then compute harmonic phonons for each deformed structure.
    Finally, compute Gibb's free energy.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    initial_relax_maker: .ForceFieldRelaxMaker | None
        Maker to relax the input structure.
    eos_relax_maker: .ForceFieldRelaxMaker | None
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
    min_length: float
        min length of the supercell that will be built
    max_length: float
        max length of the supercell that will be built
    prefer_90_degrees: bool
        if set to True, supercell algorithm will first try to find a supercell
        with 3 90 degree angles
    get_supercell_size_kwargs: dict
        kwargs that will be passed to get_supercell_size to determine supercell size

    """

    name: str = "Forcefield QHA Maker"
    initial_relax_maker: ForceFieldRelaxMaker | None = None
    eos_relax_maker: ForceFieldRelaxMaker | None = None
    phonon_maker: PhononMaker = None
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6
    pressure: float | None = None
    t_max: float | None = None
    ignore_imaginary_modes: bool = False
    skip_analysis: bool = False
    eos_type: Literal["vinet", "birch_murnaghan", "murnaghan"] = "vinet"
    analyze_free_energy_kwargs: dict = field(default_factory=dict)

    @property
    def prev_calc_dir_argname(self) -> None:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return

    @classmethod
    def from_force_field_name(
        cls,
        force_field_name: str | MLFF | dict,
        relax_initial_structure: bool = True,
        run_eos_flow: bool = True,
        **kwargs,
    ) -> Self:
        """
        Create a QHA flow from a forcefield name.

        Parameters
        ----------
        force_field_name : str or .MLFF or dict
            The name of the force field.
        relax_initial_structure: bool = True
            Whether to relax the initial structure before performing an EOS fit.
        run_eos_flow : bool = True
            Whether to perform an EOS fit.
        **kwargs
            Additional kwargs to pass to ForceFieldEosMaker

        Returns
        -------
        ForceFieldQhaMaker
        """
        kwargs.update(
            initial_relax_maker=(
                ForceFieldRelaxMaker(force_field_name=force_field_name)
                if relax_initial_structure
                else None
            ),
            eos_relax_maker=(
                ForceFieldRelaxMaker(
                    force_field_name=force_field_name,
                    relax_cell=False,
                    relax_kwargs={"fmax": 1e-5},
                )
                if run_eos_flow
                else None
            ),
        )
        phonon_maker = PhononMaker.from_force_field_name(
            force_field_name=force_field_name, relax_initial_structure=False
        )
        return cls(
            phonon_maker=phonon_maker,
            name=f"{phonon_maker.mlff.name} QHA Maker",
            **kwargs,
        )


@dataclass
class CHGNetQhaMaker(ForceFieldQhaMaker):
    """
    Perform quasi-harmonic approximation using CHGNet.

    First relax a structure using relax_maker.
    Then perform a series of deformations on the relaxed structure, and
    then compute harmonic phonons for each deformed structure.
    Finally, compute Gibb's free energy.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    initial_relax_maker: .ForceFieldRelaxMaker | None
        Maker to relax the input structure.
    eos_relax_maker: .ForceFieldRelaxMaker | None
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
    min_length: float
        min length of the supercell that will be built
    max_length: float
        max length of the supercell that will be built
    prefer_90_degrees: bool
        if set to True, supercell algorithm will first try to find a supercell
        with 3 90 degree angles
    get_supercell_size_kwargs: dict
        kwargs that will be passed to get_supercell_size to determine supercell size

    """

    name: str = "CHGNet QHA Maker"
    phonon_maker: PhononMaker = field(
        default_factory=lambda: PhononMaker(bulk_relax_maker=None)
    )
    initial_relax_maker: ForceFieldRelaxMaker | None = field(
        default_factory=lambda: ForceFieldRelaxMaker(force_field_name="CHGNet")
    )
    eos_relax_maker: ForceFieldRelaxMaker | None = field(
        default_factory=lambda: ForceFieldRelaxMaker(
            force_field_name="CHGNet", relax_cell=False, relax_kwargs={"fmax": 1e-5}
        )
    )
