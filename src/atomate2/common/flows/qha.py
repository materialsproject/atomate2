"""Define common EOS flow agnostic to electronic-structure code."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Maker

from atomate2.common.jobs.eos import PostProcessEosEnergy, apply_strain_to_structure
from atomate2.common.flows.eos import CommonEosMaker
from atomate2.common.flows.phonons import BasePhononMaker
from atomate2.common.jobs.qha import get_phonon_jobs, analyze_free_energy

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Job
    from pymatgen.core import Structure

    from atomate2.common.jobs.eos import EOSPostProcessor


@dataclass
class CommonQhaMaker(Maker):
    """
    Perform quasi-harmonic approximation.

    First relax a structure using relax_maker.
    Then perform a series of deformations on the relaxed structure, and
    then compute harmonic phonons for each deformed structure.
    Finally, compute Gibb's free energy.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    initial_relax_maker : .Maker | None
        Maker to relax the input structure, defaults to None (no initial relaxation).
    eos_relax_maker : .Maker
        Maker to relax deformed structures for the EOS fit.
    phonon_static_maker : .Maker | None
        Maker to generate statics after each relaxation, defaults to None.
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%.
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6.
    #postprocessor : .atomate2.common.jobs.EOSPostProcessor
    #    Optional postprocessing step, defaults to
    #    `atomate2.common.jobs.PostProcessEosEnergy`.
    #_store_transformation_information : .bool = False
    #    Whether to store the information about transformations. Unfortunately
    #    needed at present to handle issues with emmet and pydantic validation
    #    TODO: remove this when clash is fixed
    """

    name: str = "QHA Maker"
    initial_relax_maker: Maker = None
    eos_relax_maker: Maker = None
    static_maker: Maker = None
    phonon_maker: Maker = None
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6
    #postprocessor: EOSPostProcessor = field(default_factory=PostProcessEosEnergy)
    #_store_transformation_information: bool = False

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
        #In this way, one can easily exchange makers and enforce postprocessor None
        self.eos=CommonEosMaker(initial_relax_maker=self.initial_relax_maker, eos_relax_maker=self.eos_relax_maker,
                                static_maker=self.static_maker, postprocessor=None,
                                number_of_frames=self.number_of_frames)

        eos_job=self.eos.make(structure)
        # Todo: think about whether to keep the tight relax here
        phonon_jobs=get_phonon_jobs(self.phonon_maker, eos_job.output)

        # Todo: reuse postprocessor from equation of state to make fits of free energy curves
        # get free energy fits and perform qha
        analysis=analyze_free_energy(eos_job.output, phonon_jobs.output)

        return Flow([eos_job, phonon_jobs, analysis])