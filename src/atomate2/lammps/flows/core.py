from dataclasses import dataclass, field

from jobflow import Flow, Maker
from pymatgen.core import Structure

from ..jobs.base import LammpsMaker
from ..jobs.core import MeltMaker, QuenchMaker, ThermalizeMaker


@dataclass
class MeltQuenchThermalizeMaker(Maker):
    # potentially remove and replace with single job
    name: str = "melt-quench-thermalize"
    melt_maker: LammpsMaker = field(default_factory=MeltMaker)
    quench_maker: LammpsMaker = field(default_factory=QuenchMaker)
    thermalize_maker: LammpsMaker = field(default_factory=ThermalizeMaker)

    def make(self, structure: Structure):
        melt = self.melt_maker.make(structure)
        quench = self.quench_maker.make(melt.output.structure)
        thermalize = self.thermalize_make.make(quench.output.structure)
        return Flow([melt, quench, thermalize], name=self.name)