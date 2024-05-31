# insert makers for flows
# reference: https://github.com/hrushikesh-s/atomate2/tree/hiphive/src/atomate2/vasp/flows
# Try Double Relax

"""Core Qchem flows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from emmet.core.qchem.calculation import Calculation
from jobflow import Flow, Maker, Response, job

from atomate2.qchem.jobs.core import (  
    SinglePointMaker,
    OptMaker,
    ForceMaker,
    TransitionStateMaker,
    FreqMaker,
)
from atomate2.qchem.sets.core import (
    SinglePointSetGenerator,
    OptSetGenerator,
    ForceSetGenerator,
    TransitionStateSetGenerator,
    FreqSetGenerator,
)

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Job
    from pymatgen.core.structure import Molecule 

    from atomate2.qchem.jobs.base import BaseQCMaker

import numpy as np

@dataclass
class DoubleOptMaker(Maker):
    """
    Maker to perform a double Qchem relaxation.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to use to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to use to generate the second relaxation.
    """

    name: str = "double opt"
    opt_maker1: BaseQCMaker | None = field(default_factory=OptMaker)
    opt_maker2: BaseQCMaker = field(default_factory=OptMaker)

    def make(self, molecule: Molecule, prev_dir: str | Path | None = None) -> Flow:
        """
        Create a flow with two chained molecular optimizations.

        Parameters
        ----------
        molecule : .Molecule
            A pymatgen Molecule object.
        prev_dir : str or Path or None
            A previous QChem calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing two geometric optimizations.
        """
        jobs: list[Job] = []
        if self.opt_maker1:
            # Run a pre-relaxation
            opt1 = self.opt_maker1.make(molecule, prev_dir=prev_dir)
            opt1.name += " 1"
            jobs += [opt1]
            molecule = opt1.output.optimized_molecule
            prev_dir = opt1.output.dir_name

        opt2 = self.opt_maker2.make(molecule, prev_dir=prev_dir)
        opt2.name += " 2"
        jobs += [opt2]

        return Flow(jobs, output=opt2.output, name=self.name)

    @classmethod
    def from_opt_maker(cls, opt_maker: BaseQCMaker) -> DoubleOptMaker:
        """
        Instantiate the DoubleRelaxMaker with two relax makers of the same type.

        Parameters
        ----------
        opt_maker : .BaseQCMaker
            Maker to use to generate the first and second geometric optimizations.
        """
        return cls(
            relax_maker1=deepcopy(opt_maker), relax_maker2=deepcopy(opt_maker)
        )



@dataclass
class FrequencyOptMaker(Maker):
    """
    Maker to perform a frequency calculation after an optimization.
    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    opt_maker : .BaseQCMaker
        Maker to use to generate the opt maker
    freq_maker : .BaseQCMaker
        Maker to use to generate the freq maker
    """

    name: str = "opt frequency"
    opt_maker: BaseQCMaker = field(default_factory=OptMaker)
    freq_maker: BaseQCMaker = field(default_factory= FreqMaker)

    def make(self, molecule: Molecule, prev_dir: str | Path | None = None) -> Flow:
        """
        Create a flow with optimization followed by frequency calculation.

        Parameters
        ----------
        molecule : .Molecule
            A pymatgen Molecule object.
        prev_dir : str or Path or None
            A previous QChem calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing with optimization and frequency calculation.
        """
        jobs: list[Job] = []
        opt = self.opt_maker.make(molecule, prev_qchem_dir=prev_dir)
        #opt.name += " 1"
        opt.name = 'Geometry Optimization'
        jobs += [opt]
        opt_taskdoc = opt.output
        molecule =  opt_taskdoc.output.optimized_molecule
        #prev_dir = opt_taskdoc.dir_name


        freq = self.freq_maker.make(molecule, prev_qchem_dir=prev_dir)
        #freq.name += " 1"
        freq.name = 'Frequency Analysis'
        jobs += [freq]
        freq_taskdoc = freq.output
        modes = freq_taskdoc.output.calcs_reversed[0].output.frequency_modes
        frequencies = freq_taskdoc.output.calcs_reversed[0].output.frequencies

        return Flow(jobs, output={'opt': opt.output, 'freq':freq.output},name=self.name)


@dataclass
class FrequencyOptFlatteningMaker(Maker):
    """
    Maker to perform a frequency calculation after an optimization.
    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    opt_maker : .BaseQCMaker
        Maker to use to generate the opt maker
    freq_maker : .BaseQCMaker
        Maker to use to generate the freq maker
    """

    name: str = "frequency flattening opt"
    opt_maker: BaseQCMaker = field(default_factory=OptMaker)
    freq_maker: BaseQCMaker = field(default_factory= FreqMaker)
    scale: float = 1.0
    max_ffopt_runs: int = 5


    @job
    def make(self, molecule: Molecule, mode: list | None = None , lowest_freq: float = - 1.0, ffopt_runs: int = 0, overwrite_inputs : dict | None = None  ,prev_dir: str | Path | None = None) -> Flow:
        """
        Create a flow with optimization followed by frequency calculation with perturbation along the negative frequency mode

        Parameters
        ----------
        molecule : .Molecule
            A pymatgen Molecule object.
        prev_dir : str or Path or None
            A previous QChem calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing with optimization and frequency calculation.
        """
        if mode is None:
            mode = []
            for _ in range(len(molecule)):
                mode.append([0.0,0.0,0.0])

        if overwrite_inputs is not None:
            self.opt_maker.input_set_generator.overwrite_inputs = overwrite_inputs
            self.freq_maker.input_set_generator.overwrite_inputs = overwrite_inputs

        jobs: list[Job] = []
        opt_taskdoc = None
        freq_taskdoc = None
        if (lowest_freq < 0) and (ffopt_runs < self.max_ffopt_runs):
            molecule_copy = deepcopy(molecule)
            for ii in range(len(molecule)):
                vec = np.array(mode[ii])
                molecule_copy.translate_sites(indices=[ii], vector=vec * self.scale)
            molecule = molecule_copy
            
            opt = self.opt_maker.make(molecule, prev_qchem_dir=prev_dir)
            opt.name = 'Geometry Optimization'
            jobs += [opt]
            opt_taskdoc = opt.output
            molecule = opt_taskdoc.output.optimized_molecule
            

            freq = self.freq_maker.make(molecule, prev_qchem_dir=prev_dir)
            freq.name = 'Frequency Analysis'
            jobs += [freq]
            freq_taskdoc = freq.output
            modes = freq_taskdoc.output.frequency_modes
            frequencies = freq_taskdoc.output.frequencies
            ffopt_runs = ffopt_runs + 1
                 
            recursive = self.make(molecule,
                                  mode = modes[0],
                                  lowest_freq = frequencies[0],
                                  ffopt_runs = ffopt_runs, 
                                  prev_dir=prev_dir)
            new_flow = Flow([*jobs, recursive], output = recursive.output)
            return Response(replace = new_flow, output = recursive.output)
        else:
            freq = self.freq_maker.make(molecule, prev_qchem_dir=prev_dir)
            freq.name = 'Frequency Analysis'
            return Response(replace = [freq], output = freq.output)
