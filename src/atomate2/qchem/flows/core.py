"""Core QChem Flows."""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.qchem.jobs.base import BaseQCMaker
from atomate2.qchem.jobs.core import FreqMaker, OptMaker, TransitionStateMaker

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Molecule

# from atomate2.qchem.schemas.calculation import VaspObject
# from atomate2.vasp.sets.core import HSEBSSetGenerator, NonSCFSetGenerator

__all__ = [
    "FrequencyFlatteningOptimizeMaker",
    "FrequencyFlatteningTransitionStateMaker",
]


@dataclass
class FrequencyFlatteningOptimizeMaker(Maker):
    """
    Iterative Maker for FrequencyFlattening.

    Maker to iteratively optimize given structure and flatten
    imaginary frequencies to ensure that the resulting structure
    is a true minima and not a saddle point.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    freq_maker : .BaseQChemMaker
        Maker to generate the frequency calculation.
    opt_maker : .BaseQChemMaker
        Maker to generate the optimization calculation.

    """

    name: str = "frequency flattening structure optimization"
    max_iterations: int = 10
    max_molecule_perturb_scale: float = 0.3
    linked: bool = True
    freq_before_opt: bool = False
    perturb_geometry: bool = False
    mode = None
    scale: float = 1.0
    max_errors: int = 20

    opt_maker: BaseQCMaker = field(default_factory=OptMaker)
    freq_maker: BaseQCMaker = field(default_factory=FreqMaker)

    warnings.warn(
        "This is a Test implementation. Use previous implementations for now",
        Warning,
        stacklevel=2,
    )

    def make(self, molecule: Molecule, prev_qchem_dir: str | Path | None = None):
        """
        Create a flow with iterative frequency and optimization calculations.

        Parameters
        ----------
        molecule : .Molecule
            A pymatgen moelcule object.
        prev_qchem_dir : str or Path or None
            A previous QChem calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing two optimizations.
        """
        freq1 = self.freq_maker.make(molecule, prev_qchem_dir=prev_qchem_dir)
        freq1.name += " 1"

        opt2 = self.opt_maker.make(
            freq1.output.molecule, prev_qchem_dir=freq1.output.dir_name
        )
        opt2.name += " 2"

        return Flow([freq1, opt2], opt2.output, name=self.name)

    @classmethod
    def from_freq_and_opt_maker(cls, freq_maker: BaseQCMaker, opt_maker=BaseQCMaker):
        """
        Instantiate the FrequencyFlatteningOptimizeMaker with a Freq and an Opt maker.

        Parameters
        ----------
        freq_maker : .BaseQChemMaker
        opt_maker : .BaseQChemMaker
            Maker to use to generate the frequency flattening
            and subsequent optimization.
        """
        return cls(freq_maker_1=deepcopy(freq_maker), opt_maker_2=deepcopy(opt_maker))


@dataclass
class FrequencyFlatteningTransitionStateMaker(Maker):
    # need to incorporate iterations and perturb geometry
    """
    Iterative Maker for Transition State Freq Flattening.

    Maker to iteratively optimize transition state structure
    and flatten imaginary frequencies to ensure that
    the resulting structure is a transition state.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    freq_maker : .BaseQChemMaker
        Maker to generate the frequency calculation.
    ts_maker : .BaseQChemMaker
        Maker to generate the optimization calculation.
    """

    name: str = "frequency flattening transition state optimization"
    max_iterations: int = 10
    max_molecule_perturb_scale: float = 0.3
    linked: bool = True
    freq_before_opt: bool = False
    perturb_geometry: bool = False
    mode = None
    scale: float = 1.0
    max_errors: int = 20

    ts_maker: BaseQCMaker = field(default_factory=TransitionStateMaker)
    freq_maker: BaseQCMaker = field(default_factory=FreqMaker)

    warnings.warn(
        "This is a Test implementation. Use previous implementations for now",
        Warning,
        stacklevel=2,
    )

    def make(self, molecule: Molecule, prev_qchem_dir: str | Path | None = None):
        """
        Create a flow with iterative frequency and ts optimization calculations.

        Parameters
        ----------
        molecule : .Molecule
            A pymatgen moelcule object.
        prev_qchem_dir : str or Path or None
            A previous QChem calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing two relaxations.
        """
        freq1 = self.freq_maker1.make(molecule, prev_qchem_dir=prev_qchem_dir)
        freq1.name += " 1"

        ts2 = self.ts_maker2.make(
            freq1.output.molecule, prev_qchem_dir=freq1.output.dir_name
        )
        ts2.name += " 2"

        return Flow([freq1, ts2], ts2.output, name=self.name)

    @classmethod
    def from_freq_and_ts_maker(cls, freq_maker: BaseQCMaker, ts_maker=BaseQCMaker):
        """
        Instantiate the FrequencyFlatteningOptimizeMaker with a Freq and a TS maker.

        Parameters
        ----------
        freq_maker : .BaseQChemMaker
        ts_maker : .BaseQChemMaker
            Maker to use to generate the frequency flattening
            and subsequent optimization.
        """
        return cls(freq_maker_1=deepcopy(freq_maker), ts_maker_2=deepcopy(ts_maker))
