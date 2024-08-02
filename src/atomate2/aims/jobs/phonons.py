"""Define the PhononDisplacementMakers for FHI-aims."""

from dataclasses import dataclass, field

from pymatgen.io.aims.sets.base import AimsInputGenerator
from pymatgen.io.aims.sets.core import SocketIOSetGenerator, StaticSetGenerator

from atomate2.aims.jobs.base import BaseAimsMaker
from atomate2.aims.jobs.core import SocketIOStaticMaker


@dataclass
class PhononDisplacementMaker(BaseAimsMaker):
    """
    Maker to perform a static calculation as a part of the finite displacement method.

    The input set is for a static run with tighter convergence parameters.
    Both the k-point mesh density and convergence parameters
    are stricter than a normal relaxation.

    Parameters
    ----------
    name: str
        The job name.
    input_set_generator: .AimsInputGenerator
        A generator used to make the input set.
    """

    name: str = "phonon static aims"

    input_set_generator: AimsInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_params={"compute_forces": True},
            user_kpoints_settings={"density": 5.0, "even": True},
        )
    )


@dataclass
class PhononDisplacementMakerSocket(SocketIOStaticMaker):
    """
    Maker to perform a static calculation as a part of the finite displacement method.

    The input set is for a static run with tighter convergence parameters.
    Both the k-point mesh density and convergence parameters
    are stricter than a normal relaxation.

    Parameters
    ----------
    name: str
        The job name.
    input_set_generator: .AimsInputGenerator
        A generator used to make the input set.
    """

    name: str = "phonon static aims socket"

    input_set_generator: AimsInputGenerator = field(
        default_factory=lambda: SocketIOSetGenerator(
            user_params={"compute_forces": True},
            user_kpoints_settings={"density": 5.0, "even": True},
        )
    )
