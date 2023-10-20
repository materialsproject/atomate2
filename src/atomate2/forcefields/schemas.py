"""Job to prerelax a structure using an MD Potential."""

from typing import Optional
from ase.stress import voigt_6_to_full_3x3_stress
from emmet.core.structure import StructureMetadata
from emmet.core.math import Matrix3D, Vector3D
from pydantic import BaseModel, Extra, Field
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor


class IonicStep(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """Document defining the information at each ionic step."""

    energy: float = Field(None, description="The free energy.")
    forces: Optional[list[list[float]]] = Field(
        None, description="The forces on each atom."
    )
    stress: Optional[Matrix3D] = Field(None, description="The stress on the lattice.")
    structure: Structure = Field(None, description="The structure at this step.")


class InputDoc(BaseModel):
    """The inputs used to run this job."""

    structure: Structure = Field(None, description="The inputted structure.")
    relax_cell: bool = Field(
        None,
        description="Whether cell lattice was allowed to change during relaxation.",
    )
    steps: int = Field(
        None, description="Maximum number of steps allowed during relaxation."
    )
    relax_kwargs: Optional[dict] = Field(
        None, description="Keyword arguments that passed to the relaxer function."
    )
    optimizer_kwargs: Optional[dict] = Field(
        None, description="Keyword arguments passed to the relaxer's optimizer."
    )


class OutputDoc(BaseModel):
    """The outputs of this job."""

    structure: Structure = Field(None, description="The final, relaxed structure.")

    energy: float = Field(None, description="Total energy in units of eV.")

    energy_per_atom: float = Field(
        None,
        description="Energy per atom of the final structure in units of eV/atom.",
    )

    forces: Optional[list[Vector3D]] = Field(
        None,
        description="The force on each atom in units of eV/A for the final structure.",
    )

    # NOTE: units for stresses were converted to kbar (* -10 from standard output)
    #       to comply with MP convention
    stress: Optional[Matrix3D] = Field(
        None, description="The stress on the cell in units of kbar (in Voigt notation)."
    )

    ionic_steps: list[IonicStep] = Field(
        None, description="Step-by-step trajectory of the structural relaxation."
    )

    n_steps: int = Field(
        None, description="total number of steps needed to relax the structure."
    )


class ForceFieldTaskDocument(StructureMetadata):
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

    forcefield_name: str = Field(
        None,
        description="name of the interatomic potential used for relaxation.",
    )

    forcefield_version: str = Field(
        None,
        description="version of the interatomic potential used for relaxation.",
    )

    dir_name: Optional[str] = Field(
        None, description="Directory where the force field calculations are performed."
    )

    @classmethod
    def from_ase_compatible_result(
        cls,
        forcefield_name: str,
        result: dict,
        relax_cell: bool,
        steps: int,
        relax_kwargs: dict = None,
        optimizer_kwargs: dict = None,
        ionic_step_data: tuple = ("energy", "forces", "magmoms", "stress", "structure"),
    ) -> "ForceFieldTaskDocument":
        """
        Create a ForceFieldTaskDocument for a Task that has ASE-compatible outputs.

        Parameters
        ----------
        forcefield_name : str
            Name of the force field used.
        result : dict
            The outputted results from the task.
        relax_cell : bool
            Whether the cell shape/volume was allowed to change during the task.
        steps : int
            Maximum number of ionic steps allowed during relaxation.
        relax_kwargs : dict
            Keyword arguments that will get passed to :obj:`Relaxer.relax`.
        optimizer_kwargs : dict
            Keyword arguments that will get passed to :obj:`Relaxer()`.
        ionic_step_data : tuple
            Which data to save from each ionic step.
        """
        trajectory = result["trajectory"].__dict__

        # NOTE: units for stresses were converted to kbar (* -10 from standard output) and to 3x3 matrix
        # to comply with MP convention
        for i in range(len(trajectory["stresses"])):
            trajectory["stresses"][i] = (
                voigt_6_to_full_3x3_stress(trajectory["stresses"][i]) * -10
            )

        species = AseAtomsAdaptor.get_structure(trajectory["atoms"]).species

        input_structure = Structure(
            lattice=trajectory["cells"][0],
            coords=trajectory["atom_positions"][0],
            species=species,
            coords_are_cartesian=True,
        )

        input_doc = InputDoc(
            structure=input_structure,
            relax_cell=relax_cell,
            steps=steps,
            relax_kwargs=relax_kwargs,
            optimizer_kwargs=optimizer_kwargs,
        )

        # Workaround for cases where the ASE optimizer does not correctly limit the
        # number of steps for static calculations.
        if steps <= 1:
            steps = 1
            for key in trajectory:
                trajectory[key] = [trajectory[key][0]]
            output_structure = input_structure
        else:
            output_structure = result["final_structure"]

        final_energy = trajectory["energies"][-1]
        final_energy_per_atom = trajectory["energies"][-1] / input_structure.num_sites
        final_forces = trajectory["forces"][-1].tolist()
        final_stress = trajectory["stresses"][-1].tolist()

        n_steps = len(trajectory["energies"])

        ionic_steps = []
        for i in range(n_steps):
            cur_energy = (
                trajectory["energies"][i] if "energy" in ionic_step_data else None
            )
            cur_forces = (
                trajectory["forces"][i].tolist()
                if "forces" in ionic_step_data
                else None
            )
            cur_stress = (
                trajectory["stresses"][i].tolist()
                if "stress" in ionic_step_data
                else None
            )

            if "structure" in ionic_step_data:
                cur_structure = Structure(
                    lattice=trajectory["cells"][i],
                    coords=trajectory["atom_positions"][i],
                    species=species,
                    coords_are_cartesian=True,
                )
            else:
                cur_structure = None

            # include "magmoms" in :obj:`cur_ionic_step` if the trajectory has "magmoms"
            if "magmoms" in trajectory:
                cur_ionic_step = IonicStep(
                    energy=cur_energy,
                    forces=cur_forces,
                    magmoms=(
                        trajectory["magmoms"][i].tolist()
                        if "magmoms" in ionic_step_data
                        else None
                    ),
                    stress=cur_stress,
                    structure=cur_structure,
                )

            # otherwise do not include "magmoms" in :obj:`cur_ionic_step`
            elif "magmoms" not in trajectory:
                cur_ionic_step = IonicStep(
                    energy=cur_energy,
                    forces=cur_forces,
                    stress=cur_stress,
                    structure=cur_structure,
                )

            ionic_steps.append(cur_ionic_step)

        output_doc = OutputDoc(
            structure=output_structure,
            energy=final_energy,
            energy_per_atom=final_energy_per_atom,
            forces=final_forces,
            stress=final_stress,
            ionic_steps=ionic_steps,
            n_steps=n_steps,
        )

        if forcefield_name == "M3GNet":
            import matgl

            version = matgl.__version__
        elif forcefield_name == "CHGNet":
            import chgnet

            version = chgnet.__version__
        else:
            version = "Unknown"
        return cls.from_structure(
            meta_structure=output_structure,
            structure=output_structure,
            input=input_doc,
            output=output_doc,
            forcefield_name=forcefield_name,
            forcefield_version=version,
        )
