"""Define common QHA flow agnostic to electronic-structure code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from jobflow import Flow, Maker

from atomate2.common.flows.eos import CommonEosMaker
from atomate2.common.jobs.qha import analyze_free_energy, get_phonon_jobs

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure

    from atomate2.common.flows.phonons import BasePhononMaker
    from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker

supported_eos = frozenset(("vinet", "birch_murnaghan", "murnaghan"))


@dataclass
class CommonQhaMaker(Maker, ABC):
    """
    Use the quasi-harmonic approximation.

    First relax a structure.
    Then we scale the relaxed structure, and
    then compute harmonic phonons for each scaled
    structure with Phonopy.
    Finally, we compute the Gibb's free energy and
    other thermodynamic properties available from
    the quasi-harmonic approximation.

    Note: We do not consider electronic free energies so far.
    This might be problematic for metals (see e.g.,
    Wolverton and Zunger, Phys. Rev. B, 52, 8813 (1994).)

    Note: Magnetic Materials have never been computed with
    this workflow.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    initial_relax_maker: .ForceFieldRelaxMaker | .BaseVaspMaker | None
        Maker to relax the input structure.
    eos_relax_maker: .ForceFieldRelaxMaker | .BaseVaspMaker | None
        Maker to relax deformed structures for the EOS fit.
        The volume has to be fixed!
    phonon_displacement_maker: .ForceFieldStaticMaker | .BaseVaspMaker | None
    phonon_static_maker: .ForceFieldStaticMaker | .BaseVaspMaker | None
    phonon_maker_kwargs: dict
    linear_strain: tuple[float, float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%.
    number_of_frames: int
        Number of strain calculations to do for EOS fit, default = 6.
    t_max: float | None
        Maximum temperature until which the QHA will be performed
    pressure: float | None
        Pressure at which the QHA will be performed (default None, no pressure)
    ignore_imaginary_modes: bool
        By default, volumes where the harmonic phonon approximation shows imaginary
        will be ignored
    eos_type: str
        Equation of State type used for the fitting. Defaults to vinet.
    """

    name: str = "QHA Maker"
    initial_relax_maker: ForceFieldRelaxMaker | BaseVaspMaker | None = None
    eos_relax_maker: ForceFieldRelaxMaker | BaseVaspMaker | None = None
    phonon_displacement_maker: ForceFieldStaticMaker | BaseVaspMaker | None = None
    phonon_static_maker: ForceFieldStaticMaker | BaseVaspMaker | None = None
    phonon_maker_kwargs: dict = field(default_factory=dict)
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6
    t_max: float | None = None
    pressure: float | None = None
    ignore_imaginary_modes: bool = False
    eos_type: Literal["vinet", "birch_murnaghan", "murnaghan"] = "vinet"
    analyze_free_energy_kwargs: dict = field(default_factory=dict)
    # TODO: implement advanced handling of
    #  imaginary modes in phonon runs (i.e., fitting procedures)

    def make(self, structure: Structure, prev_dir: str | Path = None) -> Flow:
        """Run an EOS flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from.

        Returns
        -------
        .Flow, a QHA flow
        """
        if self.eos_type not in supported_eos:
            raise ValueError(
                "EOS not supported.",
                "Please choose 'vinet', 'birch_murnaghan', 'murnaghan'",
            )

        # In this way, one can easily exchange makers and enforce postprocessor None
        self.eos = CommonEosMaker(
            initial_relax_maker=self.initial_relax_maker,
            eos_relax_maker=self.eos_relax_maker,
            static_maker=None,
            postprocessor=None,
            number_of_frames=self.number_of_frames,
        )
        self.phonon_maker = self.initialize_phonon_maker(
            phonon_displacement_maker=self.phonon_displacement_maker,
            phonon_static_maker=self.phonon_static_maker,
            bulk_relax_maker=None,
            phonon_maker_kwargs=self.phonon_maker_kwargs,
        )
        eos_job = self.eos.make(structure)
        phonon_jobs = get_phonon_jobs(
            phonon_maker=self.phonon_maker, eos_output=eos_job.output
        )
        analysis = analyze_free_energy(
            phonon_jobs.output,
            structure=structure,
            t_max=self.t_max,
            pressure=self.pressure,
            ignore_imaginary_modes=self.ignore_imaginary_modes,
            eos_type=self.eos_type,
            **self.analyze_free_energy_kwargs,
        )

        return Flow([eos_job, phonon_jobs, analysis])

    @abstractmethod
    def initialize_phonon_maker(
        self,
        phonon_displacement_maker: ForceFieldStaticMaker | BaseVaspMaker | None,
        phonon_static_maker: ForceFieldStaticMaker | BaseVaspMaker | None,
        bulk_relax_maker: ForceFieldRelaxMaker | BaseVaspMaker | None,
        phonon_maker_kwargs: dict,
    ) -> BasePhononMaker | None:
        """Initialize phonon maker.

        This implementation will be different for
        any newly implemented QHAMaker.

        Parameters
        ----------
        phonon_displacement_maker: ForceFieldStaticMaker|BaseVaspMaker|None
            Maker for displacement calculations.
        phonon_static_maker: ForceFieldStaticMaker|BaseVaspMaker|None
            Maker for additional static calculations.
        bulk_relax_maker: : ForceFieldRelaxMaker|BaseVaspMaker|None
            Maker for optimization. Here: None.
        phonon_maker_kwargs: dict
            Additional keyword arguments for phonon maker.

        Returns
        -------
        .BasePhononMaker

        """

    @property
    @abstractmethod
    def prev_calc_dir_argname(self) -> str | None:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
