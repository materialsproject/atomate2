"""Schema definitions for force field tasks."""

from typing import Any, Optional

from ase.stress import voigt_6_to_full_3x3_stress
from ase.units import GPa
from emmet.core.math import Matrix3D, Vector3D
from emmet.core.structure import StructureMetadata
from emmet.core.utils import ValueEnum
from emmet.core.vasp.calculation import StoreTrajectoryOption
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor


class ForcefieldObject(ValueEnum):
    """Types of forcefield data objects."""

    TRAJECTORY = "trajectory"


class IonicStep(BaseModel, extra="allow"):  # type: ignore[call-arg]
    """Document defining the information at each ionic step."""

    energy: float = Field(None, description="The free energy.")
    forces: Optional[list[list[float]]] = Field(
        None, description="The forces on each atom."
    )
    stress: Optional[Matrix3D] = Field(None, description="The stress on the lattice.")
    structure: Optional[Structure] = Field(
        None, description="The structure at this step."
    )


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

    included_objects: Optional[list[ForcefieldObject]] = Field(
        None, description="list of forcefield objects included with this task document"
    )
    forcefield_objects: Optional[dict[ForcefieldObject, Any]] = Field(
        None, description="Forcefield objects associated with this task"
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
        store_trajectory: StoreTrajectoryOption = StoreTrajectoryOption.NO,
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

        # NOTE: units for stresses were converted from eV/Angstrom³ to kBar
        # (* -1 from standard output)
        # and to 3x3 matrix to comply with MP convention
        for i in range(len(trajectory["stresses"])):
            trajectory["stresses"][i] = voigt_6_to_full_3x3_stress(
                trajectory["stresses"][i] * -10 / GPa
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
        for idx in range(n_steps):
            energy = (
                trajectory["energies"][idx] if "energy" in ionic_step_data else None
            )
            forces = (
                trajectory["forces"][idx].tolist()
                if "forces" in ionic_step_data
                else None
            )
            stress = (
                trajectory["stresses"][idx].tolist()
                if "stress" in ionic_step_data
                else None
            )

            if "structure" in ionic_step_data:
                cur_structure = Structure(
                    lattice=trajectory["cells"][idx],
                    coords=trajectory["atom_positions"][idx],
                    species=species,
                    coords_are_cartesian=True,
                )
            else:
                cur_structure = None

            # include "magmoms" in :obj:`ionic_step` if the trajectory has "magmoms"
            magmom_data = {}
            if all("magmoms" in obj for obj in (trajectory, ionic_step_data)):
                magmom_data = {"magmoms": trajectory["magmoms"][idx].tolist()}

            ionic_step = IonicStep(
                energy=energy,
                forces=forces,
                stress=stress,
                structure=cur_structure,
                **magmom_data,
            )

            ionic_steps.append(ionic_step)

        forcefield_objects: dict[ForcefieldObject, Any] = {}
        if store_trajectory != StoreTrajectoryOption.NO:
            # For VASP calculations, the PARTIAL trajectory option removes
            # electronic step info. There is no equivalent for forcefields,
            # so we just save the same info for FULL and PARTIAL options.

            traj_steps = [
                step.model_dump(exclude=("structure",)) for step in ionic_steps
            ]
            for idx in range(n_steps):
                if trajectory.get("temperatures"):
                    traj_steps[idx]["temperature"] = trajectory.get("temperatures")[idx]
                if trajectory.get("velocities"):
                    traj_steps[idx]["velocities"] = trajectory["velocities"][
                        idx
                    ].tolist()

            traj = Trajectory.from_structures(
                [step.structure for step in ionic_steps],
                frame_properties=traj_steps,
                constant_lattice=False,
            )
            forcefield_objects[ForcefieldObject.TRAJECTORY] = traj  # type: ignore[index]

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
            included_objects=list(forcefield_objects.keys()),
            forcefield_objects=forcefield_objects,
        )
