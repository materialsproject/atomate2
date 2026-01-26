"""Schema definitions for Atomic Simulation Environment (ASE) tasks.

The following code has been taken and generalized to
generic ASE calculators from
https://github.com/materialsvirtuallab/m3gnet
The code has been released under BSD 3-Clause License
and the following copyright applies:
Copyright (c) 2022, Materials Virtual Lab.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.stress import voigt_6_to_full_3x3_stress
from ase.units import GPa
from emmet.core.math import Matrix3D, Vector3D
from emmet.core.structure import MoleculeMetadata, StructureMetadata
from emmet.core.trajectory import AtomTrajectory
from emmet.core.types.enums import StoreTrajectoryOption, TaskState, ValueEnum
from pydantic import AliasChoices, BaseModel, Field
from pymatgen.core import Molecule, Structure
from pymatgen.core.trajectory import Trajectory as PmgTrajectory
from pymatgen.entries.computed_entries import ComputedEntry

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D, Vector6D

_task_doc_translation_keys = {
    "input",
    "output",
    "ase_calculator_name",
    "dir_name",
    "included_objects",
    "objects",
    "state",
    "is_force_converged",
    "energy_downhill",
    "tags",
}


def convert_stress_from_voigt_to_symm(voigt: Vector6D) -> Matrix3D:
    """Convert Voigt representation stress in GPa to 3x3 tensor in kilobar.

    Converts stress units from eV/A³ to kBar (* -1 from standard output)
    and to 3x3 matrix to comply with MP convention.

    voigt : Vector6D
        The Voigt representation of the stress in eV/A³
    """
    return tuple(voigt_6_to_full_3x3_stress(np.array(voigt) * -10 / GPa).tolist())


class AseResult(BaseModel):
    """Schema to store outputs in AseTaskDocument."""

    final_mol_or_struct: Structure | Molecule | None = Field(
        None, description="The molecule or structure in the final trajectory frame."
    )

    final_energy: float | None = Field(
        None, description="The final total energy from the calculation."
    )

    trajectory: PmgTrajectory | AtomTrajectory | None = Field(
        None, description="The relaxation or molecular dynamics trajectory."
    )

    converged: bool | None = Field(
        None, description="Whether the ASE optimizer converged."
    )

    is_force_converged: bool | None = Field(
        None,
        description=(
            "Whether the calculation is converged with respect to interatomic forces."
        ),
    )

    energy_downhill: bool | None = Field(
        None,
        description=(
            "Whether the final trajectory frame has lower total "
            "energy than the initial frame."
        ),
    )

    dir_name: str | Path | None = Field(
        None, description="The directory where the calculation was run"
    )

    elapsed_time: float | None = Field(
        None, description="The time taken to run the ASE calculation in seconds."
    )

    def __getitem__(self, name: str) -> Any:
        """Make fields subscriptable for backwards compatibility."""
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        """Allow dict-style item assignment for backwards compatibility."""
        setattr(self, name, value)


class AseObject(ValueEnum):
    """Types of ASE data objects."""

    TRAJECTORY = "trajectory"
    IONIC_STEPS = "ionic_steps"


class AseBaseModel(BaseModel):
    """Base document class for ASE input and output."""

    mol_or_struct: Structure | Molecule | None = Field(
        None,
        description="The molecule or structure at this step.",
        validation_alias=AliasChoices("mol_or_struct", "structure", "molecule"),
    )

    @property
    def structure(self) -> Structure | None:
        """Retrieve the structure associated with this document, if applicable."""
        if isinstance(self.mol_or_struct, Structure):
            return self.mol_or_struct
        return None

    @property
    def molecule(self) -> Molecule | None:
        """Retrieve the molecule associated with this document, if applicable."""
        if isinstance(self.mol_or_struct, Molecule):
            return self.mol_or_struct
        return None


class IonicStep(AseBaseModel):
    """Document defining the information at each ionic step."""

    energy: float | None = Field(None, description="The free energy.")
    forces: list[list[float]] | None = Field(
        None, description="The forces on each atom."
    )
    stress: Matrix3D | None = Field(None, description="The stress on the lattice.")
    magmoms: list[float] | None = Field(None, description="On-site magnetic moments.")


class OutputDoc(AseBaseModel):
    """The outputs of this job."""

    energy: float | None = Field(None, description="Total energy in units of eV.")

    energy_per_atom: float | None = Field(
        None,
        description="Energy per atom of the final molecule or structure "
        "in units of eV/atom.",
    )

    forces: list[Vector3D] | None = Field(
        None,
        description=(
            "The force on each atom in units of eV/A for the final molecule "
            "or structure."
        ),
    )

    # NOTE: units for stresses were converted to kbar (* -10 from standard output)
    #       to comply with MP convention
    stress: Matrix3D | None = Field(
        None, description="The stress on the cell in units of kbar."
    )

    # NOTE: the ionic_steps can also be a dict when these are in blob storage and
    #       retrieved as objects.
    ionic_steps: list[IonicStep] | dict | None = Field(
        None, description="Step-by-step trajectory of the relaxation."
    )

    elapsed_time: float | None = Field(
        None, description="The time taken to run the ASE calculation in seconds."
    )

    n_steps: int | None = Field(
        None, description="total number of steps needed in the relaxation."
    )


class InputDoc(AseBaseModel):
    """The inputs used to run this job."""

    relax_cell: bool | None = Field(
        None,
        description="Whether cell lattice was allowed to change during relaxation.",
    )
    fix_symmetry: bool | None = Field(
        None,
        description=(
            "Whether to fix the symmetry of the atoms during relaxation. "
            "Refines the symmetry of the initial molecule or structure."
        ),
    )
    symprec: float | None = Field(
        None, description="Tolerance for symmetry finding in case of fix_symmetry."
    )
    steps: int | None = Field(
        None, description="Maximum number of steps allowed during relaxation."
    )
    relax_kwargs: dict | None = Field(
        None, description="Keyword arguments that passed to the relaxer function."
    )
    optimizer_kwargs: dict | None = Field(
        None, description="Keyword arguments passed to the relaxer's optimizer."
    )


class AseStructureTaskDoc(StructureMetadata):
    """Document containing information on structure manipulation using ASE."""

    structure: Structure = Field(
        None, description="Final output structure from the task"
    )

    input: InputDoc = Field(
        None, description="The input information used to run this job."
    )

    output: OutputDoc = Field(None, description="The output information from this job.")

    ase_calculator_name: str = Field(
        None,
        description="name of the ASE calculator used in the calculation.",
    )

    dir_name: str | None = Field(
        None, description="Directory where the ASE calculations are performed."
    )

    included_objects: list[AseObject] | None = Field(
        None, description="list of ASE objects included with this task document"
    )
    objects: dict[AseObject, Any] | None = Field(
        None, description="ASE objects associated with this task"
    )

    state: TaskState | None = Field(
        None, description="Whether the calculation completed successfully."
    )

    is_force_converged: bool | None = Field(
        None,
        description=(
            "Whether the calculation is converged with respect to interatomic forces."
        ),
    )

    energy_downhill: bool | None = Field(
        None,
        description=(
            "Whether the final trajectory frame has lower total "
            "energy than the initial frame."
        ),
    )

    tags: list[str] | None = Field(None, description="List of tags for the task.")

    entry: ComputedEntry | None = Field(
        None, description="The computed entry summarizing this calculation."
    )

    @classmethod
    def from_ase_task_doc(
        cls, ase_task_doc: AseTaskDoc, **task_document_kwargs
    ) -> AseStructureTaskDoc:
        """Create an AseStructureTaskDoc for a task that has ASE-compatible outputs.

        Parameters
        ----------
        ase_task_doc : AseTaskDoc
            Task doc for the calculation
        task_document_kwargs : dict
            Additional keyword args passed to :obj:`.AseStructureTaskDoc()`.
        """
        task_document_kwargs.update(
            {k: getattr(ase_task_doc, k) for k in _task_doc_translation_keys},
            structure=ase_task_doc.mol_or_struct,
        )
        if not task_document_kwargs.get("entry"):
            task_document_kwargs["entry"] = ComputedEntry(
                composition=ase_task_doc.mol_or_struct.composition,
                energy=ase_task_doc.output.energy,
            )
        return cls.from_structure(
            meta_structure=ase_task_doc.mol_or_struct, **task_document_kwargs
        )


class AseMoleculeTaskDoc(MoleculeMetadata):
    """Document containing information on molecule manipulation using ASE."""

    molecule: Molecule = Field(None, description="Final output molecule from the task")

    input: InputDoc = Field(
        None, description="The input information used to run this job."
    )

    output: OutputDoc = Field(None, description="The output information from this job.")

    ase_calculator_name: str = Field(
        None,
        description="name of the ASE calculator used in the calculation.",
    )

    dir_name: str | None = Field(
        None, description="Directory where the ASE calculations are performed."
    )

    included_objects: list[AseObject] | None = Field(
        None, description="list of ASE objects included with this task document"
    )
    objects: dict[AseObject, Any] | None = Field(
        None, description="ASE objects associated with this task"
    )

    state: TaskState | None = Field(
        None, description="Whether the calculation completed successfully."
    )

    is_force_converged: bool | None = Field(
        None,
        description=(
            "Whether the calculation is converged with respect to interatomic forces."
        ),
    )

    energy_downhill: bool | None = Field(
        None,
        description=(
            "Whether the total energy in the final frame "
            "is less than in the initial frame."
        ),
    )

    tags: list[str] | None = Field(None, description="List of tags for the task.")


class AseTaskDoc(AseBaseModel):
    """Document containing information on generic ASE jobs."""

    input: InputDoc = Field(
        None, description="The input information used to run this job."
    )

    output: OutputDoc = Field(None, description="The output information from this job.")

    ase_calculator_name: str = Field(
        None,
        description="name of the ASE calculator used for this job.",
    )

    dir_name: str | None = Field(
        None, description="Directory where the ASE calculations are performed."
    )

    included_objects: list[AseObject] | None = Field(
        None, description="list of ASE objects included with this task document"
    )
    objects: dict[AseObject, Any] | None = Field(
        None, description="ASE objects associated with this task"
    )

    state: TaskState | None = Field(
        None, description="Whether the calculation completed successfully."
    )

    is_force_converged: bool | None = Field(
        None,
        description=(
            "Whether the calculation is converged with respect to interatomic forces."
        ),
    )

    energy_downhill: bool | None = Field(
        None,
        description=(
            "Whether the total energy in the final frame "
            "is less than in the initial frame."
        ),
    )

    tags: list[str] | None = Field(None, description="A list of tags for the task.")

    @classmethod
    def from_ase_compatible_result(
        cls,
        ase_calculator_name: str,
        result: AseResult,
        steps: int,
        relax_kwargs: dict = None,
        optimizer_kwargs: dict = None,
        relax_cell: bool = True,
        fix_symmetry: bool = False,
        symprec: float = 1e-2,
        ionic_step_data: tuple[str, ...] | None = (
            "energy",
            "forces",
            "magmoms",
            "stress",
            "mol_or_struct",
        ),
        store_trajectory: StoreTrajectoryOption = StoreTrajectoryOption.NO,
        tags: list[str] | None = None,
        **task_document_kwargs,
    ) -> AseTaskDoc:
        """Create an AseTaskDoc for a task that has ASE-compatible outputs.

        Parameters
        ----------
        ase_calculator_name : str
            Name of the ASE calculator used.
        result : AseResult
            The output results from the task.
        steps : int
            Maximum number of ionic steps allowed during relaxation.
        relax_cell : bool = True
            Whether to allow the cell shape/volume to change during relaxation.
        fix_symmetry : bool
            Whether to fix the symmetry of the ions during relaxation.
        symprec : float
            Tolerance for symmetry finding in case of fix_symmetry.
        relax_kwargs : dict
            Keyword arguments that will get passed to :obj:`Relaxer.relax`.
        optimizer_kwargs : dict
            Keyword arguments that will get passed to :obj:`Relaxer()`.
        ionic_step_data : tuple or None
            Which data to save from each ionic step.
        store_trajectory: emmet .StoreTrajectoryOption
            Whether to store trajectory information ("no") or complete trajectories
            ("partial" or "full", which are identical).
        tags : list[str] or None
            A list of tags for the task.
        task_document_kwargs : dict
            Additional keyword args passed to :obj:`.AseTaskDoc()`.
        """
        trajectory = result.trajectory

        n_steps = None
        input_mol_or_struct = None
        if trajectory:
            n_steps = len(trajectory)
            if isinstance(trajectory, AtomTrajectory):
                input_mol_or_struct = trajectory.to_pmg(frame_props=tuple(), indices=0)[
                    0
                ]
            else:
                input_mol_or_struct = trajectory[0]

        input_doc = InputDoc(
            mol_or_struct=input_mol_or_struct,
            relax_cell=relax_cell,
            fix_symmetry=fix_symmetry,
            symprec=symprec,
            steps=steps,
            relax_kwargs=relax_kwargs,
            optimizer_kwargs=optimizer_kwargs,
        )

        # Workaround for cases where the ASE optimizer does not correctly limit the
        # number of steps for static calculations.
        if (steps is not None) and steps <= 1:
            steps = 1
            n_steps = 1

            if trajectory:
                trajectory = trajectory[-1:]
            output_mol_or_struct = input_mol_or_struct
        else:
            output_mol_or_struct = result.final_mol_or_struct

        final_energy = result.final_energy
        final_forces = None
        final_stress = None
        ionic_steps = None

        if "mol_or_struct" not in (
            user_ionic_step_data := set(ionic_step_data or tuple())
        ):
            for ms_alias in ("molecule", "structure"):
                if ms_alias in user_ionic_step_data:
                    user_ionic_step_data.add("mol_or_struct")

        if trajectory:
            ionic_step_props = {"energy", "forces"}
            if save_atoms := "mol_or_struct" in user_ionic_step_data:
                user_ionic_step_data.remove("mol_or_struct")

            if isinstance(trajectory, AtomTrajectory):
                final_energy = trajectory.energy[-1]
                final_forces = trajectory.forces[-1]

                if trajectory.stress:
                    final_stress = trajectory.stress[-1]
                    ionic_step_props.add("stress")

                if trajectory.magmoms:
                    ionic_step_props.add("magmoms")
            else:
                final_energy = trajectory.frame_properties[-1]["energy"]
                final_forces = trajectory.frame_properties[-1]["forces"]
                if any(frame.get("stress") for frame in trajectory.frame_properties):
                    final_stress = convert_stress_from_voigt_to_symm(
                        trajectory.frame_properties[-1]["stress"]
                    )
                    ionic_step_props.add("stress")

                if any(frame.get("magmoms") for frame in trajectory.frame_properties):
                    ionic_step_props.add("magmoms")

            ionic_steps = []
            use_ionic_step_props = ionic_step_props.intersection(user_ionic_step_data)
            if len(use_ionic_step_props) > 0:
                if isinstance(trajectory, AtomTrajectory):
                    ionic_steps = [
                        IonicStep(
                            mol_or_struct=trajectory.to_pmg(
                                frame_props=tuple(),
                                indices=idx,
                            )[0]
                            if save_atoms
                            else None,
                            **{
                                key: getattr(trajectory, key)[idx]
                                for key in use_ionic_step_props
                            },
                        )
                        for idx in range(n_steps)
                    ]

                else:
                    ionic_steps = [
                        IonicStep(
                            mol_or_struct=atoms if save_atoms else None,
                            **{
                                key: convert_stress_from_voigt_to_symm(
                                    trajectory.frame_properties[idx].get(key)
                                )
                                if key == "stress"
                                else trajectory.frame_properties[idx].get(key)
                                for key in use_ionic_step_props
                            },
                        )
                        for idx, atoms in enumerate(trajectory)
                    ]

        objects: dict[AseObject, Any] = {}
        if store_trajectory != StoreTrajectoryOption.NO:
            # For VASP calculations, the PARTIAL trajectory option removes
            # electronic step info. There is no equivalent for classical
            # forcefields, so we just save the same info for FULL and
            # PARTIAL options.
            objects[AseObject.TRAJECTORY] = trajectory  # type: ignore[index]

        output_doc = OutputDoc(
            mol_or_struct=output_mol_or_struct,
            energy=final_energy,
            energy_per_atom=final_energy / len(output_mol_or_struct),
            forces=final_forces,
            stress=final_stress,
            ionic_steps=ionic_steps,
            elapsed_time=result.elapsed_time,
            n_steps=n_steps,
        )

        state = None
        if result.converged is not None:
            state = TaskState.SUCCESS if result.converged else TaskState.FAILED

        return cls(
            mol_or_struct=output_mol_or_struct,
            input=input_doc,
            output=output_doc,
            ase_calculator_name=ase_calculator_name,
            included_objects=list(objects.keys()),
            objects=objects,
            state=state,
            is_force_converged=result.is_force_converged,
            energy_downhill=result.energy_downhill,
            dir_name=result.dir_name,
            tags=tags,
            **task_document_kwargs,
        )

    @classmethod
    def to_mol_or_struct_metadata_doc(
        cls,
        ase_calculator_name: str,
        result: AseResult,
        steps: int | None = None,
        **task_document_kwargs,
    ) -> AseStructureTaskDoc | AseMoleculeTaskDoc:
        """
        Get structure and molecule specific ASE task docs.

        Parameters
        ----------
        ase_calculator_name : str
            Name of the ASE calculator used.
        result : AseResult
            The output results from the task.
        steps : int
            Maximum number of ionic steps allowed during relaxation.
        task_document_kwargs : dict
            Additional keyword args passed to :obj:`.AseTaskDoc()`.

        Returns
        -------
        AseStructureTaskDoc or AseMoleculeTaskDoc depending on `self.mol_or_struct`
        """
        task_doc = cls.from_ase_compatible_result(
            ase_calculator_name, result, steps, **task_document_kwargs
        )
        kwargs = {k: getattr(task_doc, k, None) for k in _task_doc_translation_keys}
        if isinstance(task_doc.mol_or_struct, Structure):
            meta_class = AseStructureTaskDoc
            k = "structure"
            if relax_cell := getattr(task_doc, "relax_cell", None):
                kwargs.update({"relax_cell": relax_cell})
        elif isinstance(task_doc.mol_or_struct, Molecule):
            meta_class = AseMoleculeTaskDoc
            k = "molecule"
        kwargs.update({k: task_doc.mol_or_struct, f"meta_{k}": task_doc.mol_or_struct})

        return getattr(meta_class, f"from_{k}")(**kwargs)
