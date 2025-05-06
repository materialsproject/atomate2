"""GW workflows for FHI-aims with automatic convergence."""

from dataclasses import dataclass, field

from atomate2.aims.jobs.convergence import ConvergenceMaker
from atomate2.aims.jobs.core import GWMaker


@dataclass
class GWConvergenceMaker(ConvergenceMaker):
    """A maker to perform a GW workflow with automatic convergence in FHI-aims.

    Basically a .ConvergenceMaker with adjusted defaults. Employs the fact that
    GW calculations in FHI-aims scale as O(N^4) with a large prefactor, which makes
    running a DFT part for any structure negligible with respect to the GW
    postprocessing.

    Parameters
    ----------
    name : str
        A name for the job
    maker: .GWMaker
        A maker for the run
    criterion_name: str
        A name for the convergence criterion. Must be in the run results
    epsilon: float
        A difference in criterion value for subsequent runs
    convergence_field: str
        An input parameter that changes to achieve convergence
    convergence_steps: Iterable
        An iterable of the possible values for the convergence field.
        If the iterable is depleted and the convergence is not reached,
        then the job is failed
    """

    name: str = "GW convergence"
    maker: GWMaker = field(default_factory=GWMaker)
    criterion_name: str = "bandgap"
    epsilon: float = 0.1
