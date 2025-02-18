"""Define ApproxNEB jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from monty.os.path import zpath
from pymatgen.io.vasp.outputs import Chgcar

from atomate2.utils.path import strip_hostname
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import RelaxMaker
from atomate2.vasp.run import JobType
from atomate2.vasp.sets.approx_neb import ApproxNebSetGenerator

if TYPE_CHECKING:
    from atomate2.vasp.jobs.base import BaseVaspMaker
    from atomate2.vasp.sets.base import VaspInputGenerator


@dataclass
class ApproxNebHostRelaxMaker(DoubleRelaxMaker):
    """Maker to perform a double relaxation on an ApproxNEB host structure."""

    name: str = "ApproxNEB host relax"
    relax_maker1: BaseVaspMaker | None = field(
        default_factory=lambda: RelaxMaker(input_set_generator=ApproxNebSetGenerator())
    )
    relax_maker2: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(input_set_generator=ApproxNebSetGenerator())
    )


@dataclass
class ApproxNebImageRelaxMaker(RelaxMaker):
    """
    Maker to perform a double relaxation on an ApproxNEB endpoint/image structure.

    Very important here - we are doing a double relaxation in the atomate style,
    where one job maps to two VASP calculations.
    """

    name: str = "ApproxNEB image relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: ApproxNebSetGenerator(set_type="image")
    )
    run_vasp_kwargs: dict = field(
        default_factory=lambda: {
            "job_type": JobType.DOUBLE_RELAXATION,
        }
    )


def get_charge_density(prev_dir: str | Path, use_aeccar: bool = False) -> Chgcar:
    """Get charge density from a prior VASP calculation.

    Parameters
    ----------
    prev_dir : str or Path
        Path to the previous VASP calculation
    use_aeccar : bool = False
        True: use AECCAR0 and AECCAR2 (pseudo-all electron charge density)
        rather than CHGCAR (valence electron density only - False)

    Returns
    -------
    pymatgen Chgcar object
    """
    prev_dir = Path(strip_hostname(prev_dir))
    if use_aeccar:
        aeccar0 = Chgcar.from_file(zpath(str(prev_dir / "AECCAR0")))
        aeccar2 = Chgcar.from_file(zpath(str(prev_dir / "AECCAR2")))
        return aeccar0 + aeccar2
    return Chgcar.from_file(zpath(str(prev_dir / "CHGCAR")))
