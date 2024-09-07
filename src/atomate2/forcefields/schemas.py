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
from typing_extensions import Self

from atomate2.forcefields import MLFF


class ForcefieldResult(dict):
    """Schema to store outputs in ForceFieldTaskDocument."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs = {
            "final_structure": None,
            "trajectory": None,
            "is_force_converged": None,
            **kwargs,
        }
        super().__init__(**kwargs)


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
    fix_symmetry: bool = Field(
        None,
        description=(
            "Whether to fix the symmetry of the structure during relaxation. "
            "Refines the symmetry of the initial structure."
        ),
    )
    symprec: float = Field(
        None, description="Tolerance for symmetry finding in case of fix_symmetry."
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

    is_force_converged: Optional[bool] = Field(
        None,
        description=(
            "Whether the calculation is converged with respect "
            "to interatomic forces."
        ),
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
        fix_symmetry: bool = False,
        symprec: float = 1e-2,
        ionic_step_data: tuple = ("energy", "forces", "magmoms", "stress", "structure"),
        store_trajectory: StoreTrajectoryOption = StoreTrajectoryOption.NO,
        **task_document_kwargs,
    ) -> Self:
        """Create a ForceFieldTaskDocument for a Task that has ASE-compatible outputs.

        Parameters
        ----------
        forcefield_name : str
            Name of the force field used.
        result : dict
            The outputted results from the task.
        relax_cell : bool
            Whether the cell shape/volume was allowed to change during the task.
        fix_symmetry : bool
            Whether to fix the symmetry of the structure during relaxation.
        symprec : float
            Tolerance for symmetry finding in case of fix_symmetry.
        steps : int
            Maximum number of ionic steps allowed during relaxation.
        relax_kwargs : dict
            Keyword arguments that will get passed to :obj:`Relaxer.relax`.
        optimizer_kwargs : dict
            Keyword arguments that will get passed to :obj:`Relaxer()`.
        ionic_step_data : tuple
            Which data to save from each ionic step.
        store_trajectory:
            whether to set the StoreTrajectoryOption
        task_document_kwargs : dict
            Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
        """
        trajectory = result["trajectory"]

        n_steps = len(trajectory)

        # NOTE: convert stress units from eV/A³ to kBar (* -1 from standard output)
        # and to 3x3 matrix to comply with MP convention
        for idx in range(n_steps):
            if trajectory.frame_properties[idx].get("stress") is not None:
                trajectory.frame_properties[idx]["stress"] = voigt_6_to_full_3x3_stress(
                    [
                        val * -10 / GPa
                        for val in trajectory.frame_properties[idx]["stress"]
                    ]
                )

        input_structure = trajectory[0]

        input_doc = InputDoc(
            structure=input_structure,
            relax_cell=relax_cell,
            fix_symmetry=fix_symmetry,
            symprec=symprec,
            steps=steps,
            relax_kwargs=relax_kwargs,
            optimizer_kwargs=optimizer_kwargs,
        )

        # Workaround for cases where the ASE optimizer does not correctly limit the
        # number of steps for static calculations.
        if steps <= 1:
            steps = 1
            n_steps = 1
            trajectory = Trajectory.from_structures(
                structures=[trajectory[0]],
                frame_properties=[trajectory.frame_properties[0]],
                constant_lattice=False,
            )
            output_structure = input_structure
        else:
            output_structure = result["final_structure"]

        final_energy = trajectory.frame_properties[-1]["energy"]
        final_energy_per_atom = final_energy / input_structure.num_sites
        final_forces = trajectory.frame_properties[-1]["forces"]
        final_stress = trajectory.frame_properties[-1]["stress"]

        ionic_steps = []
        for idx in range(n_steps):
            _ionic_step_data = {
                key: trajectory.frame_properties[idx][key]
                if key in ionic_step_data
                else None
                for key in ("energy", "forces", "stress")
            }

            cur_structure = trajectory[idx] if "structure" in ionic_step_data else None

            # include "magmoms" in :obj:`ionic_step` if the trajectory has "magmoms"
            if "magmoms" in trajectory.frame_properties[idx]:
                _ionic_step_data.update(
                    magmoms=trajectory.frame_properties[idx]["magmoms"]
                    if "magmoms" in ionic_step_data
                    else None
                )

            ionic_step = IonicStep(
                structure=cur_structure,
                **_ionic_step_data,
            )

            ionic_steps.append(ionic_step)

        forcefield_objects: dict[ForcefieldObject, Any] = {}
        if store_trajectory != StoreTrajectoryOption.NO:
            # For VASP calculations, the PARTIAL trajectory option removes
            # electronic step info. There is no equivalent for forcefields,
            # so we just save the same info for FULL and PARTIAL options.
            forcefield_objects[ForcefieldObject.TRAJECTORY] = trajectory  # type: ignore[index]

        output_doc = OutputDoc(
            structure=output_structure,
            energy=final_energy,
            energy_per_atom=final_energy_per_atom,
            forces=final_forces,
            stress=final_stress,
            ionic_steps=ionic_steps,
            n_steps=n_steps,
        )

        # map force field name to its package name
        model_to_pkg_map = {
            MLFF.M3GNet: "matgl",
            MLFF.CHGNet: "chgnet",
            MLFF.MACE: "mace-torch",
            MLFF.GAP: "quippy-ase",
            MLFF.Nequip: "nequip",
        }
        pkg_name = {str(k): v for k, v in model_to_pkg_map.items()}.get(forcefield_name)
        if pkg_name:
            import importlib.metadata

            version = importlib.metadata.version(pkg_name)
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
            is_force_converged=result.get("is_force_converged"),
            **task_document_kwargs,
        )
