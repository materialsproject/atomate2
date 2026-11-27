"""Schemas for Quantum ESPRESSO calculations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io.espresso import read_espresso_out, read_espresso_ph, read_fortran_namelist
from emmet.core.math import Matrix3D
from emmet.core.structure import StructureMetadata
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.ase.schemas import OutputDoc, convert_stress_from_voigt_to_symm


def _read_namelists(path: Path) -> dict:
    """Read the namelists of a Quantum ESPRESSO input file as dictionaries."""
    with open(path) as file:
        namelists, _ = read_fortran_namelist(file)
    return {name: dict(values) for name, values in namelists.items() if values}


class EspressoInput(BaseModel):
    """The input of a Quantum ESPRESSO calculation."""

    structure: Structure = Field(description="The input structure.")
    parameters: dict = Field(description="The pw.x namelists of the input file.")
    ph_parameters: dict | None = Field(
        None, description="The ph.x namelists of the input file, if ph.x was run."
    )


class EspressoOutput(OutputDoc):
    """The output of a Quantum ESPRESSO calculation."""

    epsilon_static: Matrix3D | None = Field(
        None, description="The high-frequency dielectric tensor from ph.x."
    )
    born: list[Matrix3D] | None = Field(
        None,
        description="The Born effective charges from ph.x (d Force / dE), with the "
        "acoustic sum rule applied by ph.x.",
    )


class EspressoTaskDoc(StructureMetadata):
    """A document with the input and the output of a Quantum ESPRESSO calculation."""

    dir_name: str = Field(description="The directory of the calculation.")
    task_label: str | None = Field(None, description="The name of the job.")
    structure: Structure = Field(description="The final structure.")
    input: EspressoInput = Field(description="The input of the calculation.")
    output: EspressoOutput = Field(description="The output of the calculation.")

    @classmethod
    def from_directory(
        cls, dir_name: str | Path, task_label: str | None = None
    ) -> EspressoTaskDoc:
        """
        Create a task document from the files of a calculation directory.

        The directory has the pw.x input and output files "pw.in" and "pw.out",
        and for a dielectric calculation the ph.x files "ph.in" and "ph.out".

        Parameters
        ----------
        dir_name : str or Path
            The directory of the calculation.
        task_label : str or None
            The name of the job.

        Returns
        -------
        EspressoTaskDoc
            The task document.
        """
        dir_name = Path(dir_name)
        parameters = _read_namelists(dir_name / "pw.in")
        with open(dir_name / "pw.out") as file:
            steps = list(read_espresso_out(file, index=slice(None)))
        initial, final = steps[0], steps[-1]
        structure = AseAtomsAdaptor.get_structure(final)

        ph_parameters = epsilon_static = born = None
        if (dir_name / "ph.out").exists():
            ph_parameters = _read_namelists(dir_name / "ph.in")
            with open(dir_name / "ph.out") as file:
                gamma = next(iter(read_espresso_ph(file).values()))
            epsilon_static = np.array(gamma["dieltensor"]).tolist()
            born = np.array(gamma["borneffcharge"]).tolist()

        energy = final.get_potential_energy()
        forces = final.calc.results.get("forces")
        stress = final.calc.results.get("stress")
        if stress is not None:
            stress = convert_stress_from_voigt_to_symm(stress)
        return cls.from_structure(
            meta_structure=structure,
            dir_name=str(dir_name),
            task_label=task_label,
            structure=structure,
            input=EspressoInput(
                structure=AseAtomsAdaptor.get_structure(initial),
                parameters=parameters,
                ph_parameters=ph_parameters,
            ),
            output=EspressoOutput(
                energy=energy,
                energy_per_atom=energy / len(structure),
                mol_or_struct=structure,
                forces=None if forces is None else forces.tolist(),
                stress=stress,
                epsilon_static=epsilon_static,
                born=born,
            ),
        )
