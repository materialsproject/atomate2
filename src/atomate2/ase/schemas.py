"""Schema definitions for Atomic Simulation Environment (ASE) tasks.

The following code has been taken and generalized to
generic ASE calculators from
https://github.com/materialsvirtuallab/m3gnet
The code has been released under BSD 3-Clause License
and the following copyright applies:
Copyright (c) 2022, Materials Virtual Lab.
"""

from __future__ import annotations

from typing import Any, Optional

from ase.stress import voigt_6_to_full_3x3_stress
from ase.units import GPa
from emmet.core.math import Matrix3D, Vector3D
from emmet.core.structure import StructureMetadata
from emmet.core.utils import ValueEnum
from emmet.core.vasp.calculation import StoreTrajectoryOption
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.core.trajectory import Trajectory as PmgTrajectory


class AseResult(dict):
    """Schema to store outputs in ForceFieldTaskDocument."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs = {
            "final_structure": None,
            "trajectory": None,
            "is_force_converged": None,
            "energy_downhill": None,
            **kwargs,
        }
        super().__init__(**kwargs)


class AseObject(ValueEnum):
    """Types of ASE data objects."""

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


class InputDoc(BaseModel):
    """The inputs used to run this job."""

    structure: Structure = Field(None, description="The input structure.")
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


class AseTaskDocument(StructureMetadata):
    """Document containing information on structure manipulation using ASE."""

    structure: Structure = Field(
        None, description="Final output structure from the task"
    )

    input: InputDoc = Field(
        None, description="The input information used to run this job."
    )

    output: OutputDoc = Field(
        None, description="The output information from this relaxation job."
    )

    ase_calculator_name: str = Field(
        None,
        description="name of the ASE calculator used for relaxation.",
    )

    dir_name: Optional[str] = Field(
        None, description="Directory where the ASE calculations are performed."
    )

    included_objects: Optional[list[AseObject]] = Field(
        None, description="list of ASE objects included with this task document"
    )
    objects: Optional[dict[AseObject, Any]] = Field(
        None, description="ASE objects associated with this task"
    )

    is_force_converged: Optional[bool] = Field(
        None,
        description=(
            "Whether the calculation is converged with respect "
            "to interatomic forces."
        ),
    )

    energy_downhill : Optional[bool] = Field(
        None, description="Whether the total energy in the final frame is less than in the initial frame."
    )

    @classmethod
    def from_ase_compatible_result(
        cls,
        ase_calculator_name: str,
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
    ) -> AseTaskDocument:
        """Create an AseTaskDocument for a Task that has ASE-compatible outputs.

        Parameters
        ----------
        ase_calculator_name : str
            Name of the ASE calculator used.
        result : dict
            The output results from the task.
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
            trajectory = PmgTrajectory.from_structures(
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
                    {
                        "magmoms": trajectory.frame_properties[idx]["magmoms"]
                        if "magmoms" in ionic_step_data
                        else None
                    }
                )

            ionic_step = IonicStep(
                structure=cur_structure,
                **_ionic_step_data,
            )

            ionic_steps.append(ionic_step)

        objects: dict[AseObject, Any] = {}
        if store_trajectory != StoreTrajectoryOption.NO:
            # For VASP calculations, the PARTIAL trajectory option removes
            # electronic step info. There is no equivalent for classical
            # forcefields, so we just save the same info for FULL and
            # PARTIAL options.
            objects[AseObject.TRAJECTORY] = trajectory  # type: ignore[index]

        output_doc = OutputDoc(
            structure=output_structure,
            energy=final_energy,
            energy_per_atom=final_energy_per_atom,
            forces=final_forces,
            stress=final_stress,
            ionic_steps=ionic_steps,
            n_steps=n_steps,
        )

        return cls.from_structure(
            meta_structure=output_structure,
            structure=output_structure,
            input=input_doc,
            output=output_doc,
            ase_calculator_name=ase_calculator_name,
            included_objects=list(objects.keys()),
            objects=objects,
            is_force_converged=result.get("is_force_converged"),
            energy_downhill = result.get("energy_downhill"),
            **task_document_kwargs,
        )
