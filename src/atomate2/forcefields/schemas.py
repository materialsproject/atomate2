"""Job to prerelax a structure using an MD Potential."""

import logging
from typing import List

from pydantic import BaseModel, Field

# from jobflow import Maker, job
from pymatgen.core.structure import Structure


class InputDoc(BaseModel):
    """The inputs used to run this job."""

    structure: Structure = Field(None, description="The inputted structure.")
    relax_cell: bool = Field(
        None,
        description="Whether cell lattice was allowed to change during relaxation.",
    )
    relax_kwargs: dict = Field(
        None,
        description="Keyword arguments that passed to :obj:`StructOptimizer.relax`.",
    )
    optimizer_kwargs: dict = Field(
        None,
        description="Keyword arguments that passed to :obj:`StructOptimizer`.",
    )


class OutputDoc(BaseModel):
    """The outputs of this job."""

    structure: Structure = Field(None, description="The final, relaxed structure.")

    energy: float = Field(None, description="Total energy in units of eV.")

    energy_per_atom: float = Field(
        None,
        description="Energy per atom of the final structure in units of eV/atom.",
    )

    forces: List[List[float]] = Field(
        None,
        description="The force on each atom in units of eV/A for the final structure.",
    )

    # NOTE: units for stresses were converted to kbar
    # (* -10 from standard output) to comply with MP convention
    stress: List[float] = Field(
        None, description="The stress on the cell in units of kbar (in Voigt notation)."
    )

    trajectory: dict = Field(
        None, description="Step-by-step trajectory of the structural relaxation."
    )

    steps: int = Field(
        None, description="total number of steps needed to relax the structure."
    )


class FFStructureRelaxDocument(BaseModel):
    """Document containing information on structure relaxation using a force field."""

    structure: Structure = Field(
        None, description="Final output structure from the task"
    )

    input: InputDoc = Field(
        None, description="The inputted information used to run this job."
    )

    output: OutputDoc = Field(
        None, description="The outputted information from this relaxation job."
    )

    md_potential: str = Field(
        None,
        description="name of the interatomic potential used for relaxation.",
    )

    MD_potential_version: str = Field(
        None,
        description="version of the interatomic potential used for relaxation.",
    )

    @classmethod
    def from_chgnet_result(
        cls,
        input_structure: Structure,
        relax_cell: bool,
        relax_kwargs: dict,
        optimizer_kwargs: dict,
        result: dict,
        keep_info: list,
    ):
        """
        Create a FFStructureRelaxDocument for a CHGNet relaxation.

        Parameters
        ----------
        input_structure : .Structure
            The inputted pymatgen structure.
        relax_cell: bool
            Whether the cell shape/volume was allowed to change during pre-relaxation.
        relax_kwargs : dict
            Keyword arguments that will get passed to :obj:`StructOptimizer.relax`.
        optimizer_kwargs : dict
            Keyword arguments that will get passed to :obj:`StructOptimizer()`.
        result : dict
            The outputted results from relaxation.
        keep_info : list
            List of which pieces of information from the trajectory is saved in this
            FFStructureRelaxDocument.
        """
        input_doc = InputDoc(
            structure=input_structure,
            relax_cell=relax_cell,
            relax_kwargs=relax_kwargs,
            optimizer_kwargs=optimizer_kwargs,
        )

        output_structure = result["final_structure"]
        trajectory = result["trajectory"].__dict__
        # NOTE: units for stresses were converted to kbar (* -10 from standard output)
        # to comply with MP convention
        for i in range(0, len(trajectory["stresses"])):
            trajectory["stresses"][i] = trajectory["stresses"][i] * -10
        final_energy = trajectory["energies"][-1]
        final_energy_per_atom = trajectory["energies"][-1] / output_structure.num_sites
        final_forces = trajectory["forces"][-1].tolist()
        final_stress = trajectory["stresses"][-1].tolist()

        trajectory_to_save = {key: trajectory[key] for key in keep_info}
        if "atoms" in keep_info:
            warning_msg = """
                WARNING: `Atoms` objects can't be serialized (as of May 2023) and
                thus are automatically removed.
                """
            logging.warning(warning_msg)
            trajectory_to_save.pop(
                "atoms"
            )  # can't serialize `AseAtoms` objects, so remove them from trajectory.

        steps = len(result["trajectory"])

        output_doc = OutputDoc(
            structure=output_structure,
            energy=final_energy,
            energy_per_atom=final_energy_per_atom,
            forces=final_forces,
            stress=final_stress,
            trajectory=trajectory_to_save,
            steps=steps,
        )

        import chgnet

        version = chgnet.__version__

        return cls(
            structure=output_structure,
            input=input_doc,
            output=output_doc,
            MD_potential="CHGNet",
            MD_potential_version=version,
        )
