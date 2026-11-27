"""Job makers for Quantum ESPRESSO calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import Maker, job
from pymatgen.util.due import Doi, due

from atomate2 import SETTINGS
from atomate2.espresso.run import run_espresso
from atomate2.espresso.schemas import EspressoTaskDoc
from atomate2.espresso.sets import (
    EspressoDielectricInputGenerator,
    EspressoInputGenerator,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure


@due.dcite(Doi("10.1088/1361-648X/aa8f79"), description="Quantum ESPRESSO")
@dataclass
class BaseEspressoMaker(Maker):
    """
    Base maker of Quantum ESPRESSO jobs.

    The job writes the inputs of the input set generator, runs pw.x with
    ``SETTINGS.ESPRESSO_PW_CMD``, and, if the input set has a ph.x input, runs
    ph.x with ``SETTINGS.ESPRESSO_PH_CMD``. pw.x can stop with exit code 0 when
    the self-consistency or the relaxation did not converge, so the job fails if
    pw.out shows either.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .EspressoInputGenerator
        The generator of the input set.
    """

    name: str = "base espresso job"
    input_set_generator: EspressoInputGenerator = field(
        default_factory=EspressoInputGenerator
    )

    @job(output_schema=EspressoTaskDoc)
    def make(
        self, structure: Structure, prev_dir: str | Path | None = None
    ) -> EspressoTaskDoc:
        """
        Run a Quantum ESPRESSO calculation.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure.
        prev_dir : str or Path or None
            Not used. It matches the signature of the makers of other codes.

        Returns
        -------
        EspressoTaskDoc
            The task document of the calculation.
        """
        input_set = self.input_set_generator.get_input_set(structure)
        input_set.write_input(".")
        run_espresso(SETTINGS.ESPRESSO_PW_CMD, "pw.in", "pw.out")
        pw_output = Path("pw.out").read_text()
        if "convergence NOT achieved" in pw_output:
            raise RuntimeError(f"pw.x did not converge, see {Path.cwd() / 'pw.out'}")
        if "ph.in" in input_set:
            run_espresso(SETTINGS.ESPRESSO_PH_CMD, "ph.in", "ph.out")

        task_doc = EspressoTaskDoc.from_directory(Path.cwd(), task_label=self.name)
        calculation = task_doc.input.parameters["control"]["calculation"]
        # BFGS prints "bfgs converged", damped dynamics and FIRE print
        # "convergence achieved"
        relaxed = "bfgs converged" in pw_output or "convergence achieved" in pw_output
        if calculation in {"relax", "vc-relax"} and not relaxed:
            raise RuntimeError(f"pw.x did not relax, see {Path.cwd() / 'pw.out'}")
        return task_doc


@dataclass
class EspressoStaticMaker(BaseEspressoMaker):
    """
    Maker of a self-consistent pw.x calculation of the energy, forces, and stress.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .EspressoInputGenerator
        The generator of the input set.
    """

    name: str = "espresso static"


@dataclass
class EspressoRelaxMaker(BaseEspressoMaker):
    """
    Maker of a pw.x relaxation of the atomic positions and the cell.

    The relaxation stops when the energy change is below 1e-6 Ry, the forces are
    below 1e-4 Ry/Bohr (2.6 meV/Å), and the pressure is below 0.1 kbar. The
    output is from the final self-consistent calculation of pw.x on the relaxed
    cell.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .EspressoInputGenerator
        The generator of the input set.
    """

    name: str = "espresso relax"
    input_set_generator: EspressoInputGenerator = field(
        default_factory=lambda: EspressoInputGenerator(calculation="vc-relax")
    )


@dataclass
class EspressoDielectricMaker(BaseEspressoMaker):
    """
    Maker of the dielectric tensor and the Born effective charges with ph.x.

    These are the inputs of the non-analytical term correction in phonon
    calculations. ph.x computes them only for insulators, so pw.x runs with fixed
    occupations and without the "magmom" site property.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .EspressoDielectricInputGenerator
        The generator of the input set.
    """

    name: str = "espresso dielectric"
    input_set_generator: EspressoDielectricInputGenerator = field(
        default_factory=EspressoDielectricInputGenerator
    )
