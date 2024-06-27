from __future__ import annotations

from atomate2.common.flows.qha import CommonQhaMaker


from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.common.flows.eos import CommonEosMaker
from atomate2.forcefields.jobs import CHGNetRelaxMaker, M3GNetRelaxMaker, MACERelaxMaker
from atomate2.forcefields.flows.phonons import PhononMaker
if TYPE_CHECKING:
    from pathlib import Path
    from jobflow import Maker, Flow
    from pymatgen.core.structure import Structure



@dataclass
class CHGNetQhaMaker(CommonQhaMaker):
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

    name: str = "CHGNet QHA Maker"
    initial_relax_maker: Maker = field(default_factory=CHGNetRelaxMaker)
    # TODO understand why inheritance does not work here?
    eos_relax_maker: Maker = field(default_factory=lambda: CHGNetRelaxMaker(relax_cell=False))
    static_maker: Maker = None
    phonon_maker:  Maker = field(default_factory=lambda: PhononMaker(bulk_relax_maker=CHGNetRelaxMaker(relax_cell=False, relax_kwargs={"fmax": 0.00001}) ))
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6
    # postprocessor: EOSPostProcessor = field(default_factory=PostProcessEosEnergy)
    # _store_transformation_information: bool = False
