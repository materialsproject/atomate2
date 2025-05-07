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
from typing import Any

from ase.stress import voigt_6_to_full_3x3_stress
from ase.units import GPa
from emmet.core.math import Matrix3D, Vector3D
from emmet.core.structure import MoleculeMetadata, StructureMetadata
from emmet.core.utils import ValueEnum
from emmet.core.vasp.calculation import StoreTrajectoryOption
from pydantic import BaseModel, Field
from pymatgen.core import Molecule, Structure
from pymatgen.core.trajectory import Trajectory as PmgTrajectory

_task_doc_translation_keys = {
    "input",
    "output",
    "ase_calculator_name",
    "dir_name",
    "included_objects",
    "objects",
    "is_force_converged",
    "energy_downhill",
    "tags",
}


class AseResult(BaseModel):
    """Schema to store outputs in AseTaskDocument."""

    final_mol_or_struct: Structure | Molecule | None = Field(
        None, description="The molecule or structure in the final trajectory frame."
    )

    final_energy: float | None = Field(
        None, description="The final total energy from the calculation."
    )

    trajectory: PmgTrajectory | None = Field(
        None, description="The relaxation or molecular dynamics trajectory."
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

    dir_name: str | Path | list[str]|list[Path] |None = Field(
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

    mol_or_struct: Structure | Molecule|list[Structure]|list[Molecule] | None = Field(
        None, description="The molecule or structure at this step."
    )
    structure: Structure|list[Structure] | None = Field(None, description="The structure at this step.")
    molecule: Molecule|list[Molecule] | None = Field(None, description="The molecule at this step.")

    def model_post_init(self, _context: Any) -> None:
        """Establish alias to structure and molecule fields."""
        if self.structure is None and (isinstance(self.mol_or_struct, Structure) or isinstance(self.mol_or_struct[0], Structure)):
            self.structure = self.mol_or_struct
        elif self.molecule is None and (isinstance(self.mol_or_struct, Molecule) or isinstance(self.mol_or_struct[0], Molecule)):
            self.molecule = self.mol_or_struct


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

    energy: float|list[float] | None = Field(None, description="Total energy in units of eV.")

    energy_per_atom: float|list[float] | None = Field(
        None,
        description="Energy per atom of the final molecule or structure "
        "in units of eV/atom.",
    )

    forces: list[Vector3D]|list[list[Vector3D]] | None = Field(
        None,
        description=(
            "The force on each atom in units of eV/A for the final molecule "
            "or structure."
        ),
    )

    # NOTE: units for stresses were converted to kbar (* -10 from standard output)
    #       to comply with MP convention
    stress: Matrix3D | list[Matrix3D] |None = Field(
        None, description="The stress on the cell in units of kbar (in Voigt notation)."
    )

    # NOTE: the ionic_steps can also be a dict when these are in blob storage and
    #       retrieved as objects.
    ionic_steps: list[IonicStep]|list[list[IonicStep]] | dict|list[dict] | None = Field(
        None, description="Step-by-step trajectory of the relaxation."
    )

    elapsed_time: float | list[float]|None = Field(
        None, description="The time taken to run the ASE calculation in seconds."
    )

    n_steps: int | list[int]|None = Field(
        None, description="total number of steps needed in the relaxation."
    )

    all_forces: list[list[Vector3D]] | None = Field(
        None,
        description=(
            "The force on each atom in units of eV/A for the final molecules "
            "or structures. Only present for batch calculations."
        ),
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

    dir_name: str |list[str] |None = Field(
        None, description="Directory where the ASE calculations are performed."
    )

    included_objects: list[AseObject] | None = Field(
        None, description="list of ASE objects included with this task document"
    )
    # TODO: check if it needs to be a list
    objects: dict[AseObject, Any] | list[dict[AseObject, Any]]|None = Field(
        None, description="ASE objects associated with this task"
    )

    is_force_converged: bool | list[bool] |None = Field(
        None,
        description=(
            "Whether the calculation is converged with respect to interatomic forces."
        ),
    )

    energy_downhill: bool|list[bool] | None = Field(
        None,
        description=(
            "Whether the final trajectory frame has lower total "
            "energy than the initial frame."
        ),
    )

    tags: list[str] | None = Field(None, description="List of tags for the task.")

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
        return cls.from_structure(
            meta_structure=ase_task_doc.mol_or_struct, **task_document_kwargs
        )


class AseMoleculeTaskDoc(MoleculeMetadata):
    """Document containing information on molecule manipulation using ASE."""

    molecule: Molecule|list[Molecule] = Field(None, description="Final output molecule from the task")

    input: InputDoc = Field(
        None, description="The input information used to run this job."
    )

    output: OutputDoc = Field(None, description="The output information from this job.")

    ase_calculator_name:  str = Field(
        None,
        description="name of the ASE calculator used in the calculation.",
    )

    dir_name: str | list[str]|None = Field(
        None, description="Directory where the ASE calculations are performed."
    )

    included_objects: list[AseObject] | None = Field(
        None, description="list of ASE objects included with this task document"
    )
    objects: dict[AseObject, Any] | None = Field(
        None, description="ASE objects associated with this task"
    )

    is_force_converged: bool|list[bool] | None = Field(
        None,
        description=(
            "Whether the calculation is converged with respect to interatomic forces."
        ),
    )

    energy_downhill: bool|list[bool] | None = Field(
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

    dir_name: str |list[str]| None = Field(
        None, description="Directory where the ASE calculations are performed."
    )

    included_objects: list[AseObject] | None = Field(
        None, description="list of ASE objects included with this task document"
    )
    # maybe list
    objects: dict[AseObject, Any] |list[dict[AseObject, Any]] | None = Field(
        None, description="ASE objects associated with this task"
    )

    is_force_converged: bool | list[bool] |None = Field(
        None,
        description=(
            "Whether the calculation is converged with respect to interatomic forces."
        ),
    )

    energy_downhill: bool|list[bool] | None = Field(
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
        result: AseResult|list[AseResult],
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
        result : AseResult|list[AseResult]
            The output results from the task. Can be a list for batch jobs.
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
        is_list = not isinstance(result, AseResult)

        results= result if is_list else [result]

        output_mol_or_struct=[]
        input_mol_or_struct=[]
        final_energy=[]
        final_forces=[]
        final_stress=[]
        ionic_steps =[]
        n_steps = []
        objects =[]

        for result in results:
            trajectory = result.trajectory

            # TODO: fix this
            n_steps_here = None
            input_mol_or_struct_here = None
            if trajectory:
                n_steps_here=len(trajectory)

            # NOTE: convert stress units from eV/A³ to kBar (* -1 from standard output)
            # and to 3x3 matrix to comply with MP convention
            if n_steps_here:
                for idx in range(n_steps_here):
                    if trajectory.frame_properties[idx].get("stress") is not None:
                        trajectory.frame_properties[idx]["stress"] = (
                            voigt_6_to_full_3x3_stress(
                                [
                                    val * -10 / GPa
                                    for val in trajectory.frame_properties[idx]["stress"]
                                ]
                            )
                        )

                input_mol_or_struct_here=trajectory[0]

            input_mol_or_struct.append(input_mol_or_struct_here)
            n_steps.append(n_steps_here)

            # Workaround for cases where the ASE optimizer does not correctly limit the
            # number of steps for static calculations.
            if (steps is not None) and steps <= 1:
                steps = 1
                n_steps_here = 1

                if isinstance(input_mol_or_struct_here, Structure):
                    traj_method = "from_structures"
                elif isinstance(input_mol_or_struct_here, Molecule):
                    traj_method = "from_molecules"
                trajectory=getattr(PmgTrajectory, traj_method)(
                    [input_mol_or_struct_here],
                    frame_properties=[trajectory.frame_properties[0]],
                    constant_lattice=False,
                )
                output_mol_or_struct.append(input_mol_or_struct_here)
            else:
                output_mol_or_struct.append(result.final_mol_or_struct)

            if trajectory is None:
                final_energy.append(result.final_energy)
                final_forces.append(None)
                final_stress.append(None)
                ionic_steps.append(None)

            else:
                final_energy.append(trajectory.frame_properties[-1]["energy"])
                final_forces.append(trajectory.frame_properties[-1]["forces"])
                final_stress.append(trajectory.frame_properties[-1].get("stress"))

                ionic_steps_structure = []
                if ionic_step_data is not None and len(ionic_step_data) > 0:
                    for idx in range(n_steps_here):
                        _ionic_step_data = {
                            key: (
                                trajectory.frame_properties[idx].get(key)
                                if key in ionic_step_data
                                else None
                            )
                            for key in ("energy", "forces", "stress")
                        }

                        current_mol_or_struct = (
                            trajectory[idx]
                            if any(
                                v in ionic_step_data
                                for v in ("mol_or_struct", "structure", "molecule")
                            )
                            else None
                        )

                        # include "magmoms" in `ionic_step` if the trajectory has "magmoms"
                        if "magmoms" in trajectory.frame_properties[idx]:
                            _ionic_step_data.update(
                                {
                                    "magmoms": (
                                        trajectory.frame_properties[idx]["magmoms"]
                                        if "magmoms" in ionic_step_data
                                        else None
                                    )
                                }
                            )

                        ionic_step = IonicStep(
                            mol_or_struct=current_mol_or_struct,
                            **_ionic_step_data,
                        )

                        ionic_steps_structure.append(ionic_step)
                ionic_steps.append(ionic_steps_structure)

            objects_structure: dict[AseObject, Any] = {}
            if store_trajectory != StoreTrajectoryOption.NO:
                # For VASP calculations, the PARTIAL trajectory option removes
                # electronic step info. There is no equivalent for classical
                # forcefields, so we just save the same info for FULL and
                # PARTIAL options.
                objects_structure[AseObject.TRAJECTORY] = trajectory  # type: ignore[index]
            objects.append(objects_structure)
        if not is_list:

            input_doc = InputDoc(
                mol_or_struct=input_mol_or_struct[0],
                relax_cell=relax_cell,
                fix_symmetry=fix_symmetry,
                symprec=symprec,
                steps=steps,
                relax_kwargs=relax_kwargs,
                optimizer_kwargs=optimizer_kwargs,
            )
            output_doc = OutputDoc(
                mol_or_struct=output_mol_or_struct[0],
                energy=final_energy[0],
                energy_per_atom=final_energy[0] / len(output_mol_or_struct[0]),
                forces=final_forces[0],
                stress=final_stress[0],
                ionic_steps=ionic_steps[0],
                elapsed_time=results[0].elapsed_time,
                n_steps=n_steps[0],
            )

            return cls(
                mol_or_struct=output_mol_or_struct[0],
                input=input_doc,
                output=output_doc,
                ase_calculator_name=ase_calculator_name,
                included_objects=list(objects[0].keys()),
                objects=objects[0],
                is_force_converged=results[0].is_force_converged,
                energy_downhill=results[0].energy_downhill,
                dir_name=results[0].dir_name,
                tags=tags,
                **task_document_kwargs,
            )
        else:
            input_doc = InputDoc(
                mol_or_struct=input_mol_or_struct,
                relax_cell=relax_cell,
                fix_symmetry=fix_symmetry,
                symprec=symprec,
                steps=steps,
                relax_kwargs=relax_kwargs,
                optimizer_kwargs=optimizer_kwargs,
            )
            output_doc = OutputDoc(
                mol_or_struct=output_mol_or_struct,
                energy=final_energy,
                energy_per_atom=[final_energy_here/len(output_mol_or_struct_here) for final_energy_here, output_mol_or_struct_here  in zip(final_energy, output_mol_or_struct)],
                forces=final_forces,
                stress=final_stress,
                ionic_steps=ionic_steps,
                elapsed_time=[result.elapsed_time for result in  results],
                n_steps=n_steps,
                all_forces=final_forces,
            )

            return cls(
                mol_or_struct=output_mol_or_struct[-1], # last structure by default
                input=input_doc,
                output=output_doc,
                ase_calculator_name=ase_calculator_name,
                included_objects=list(objects[0].keys()),
                objects=objects,
                is_force_converged=[result.is_force_converged for result in results],
                energy_downhill=[result.energy_downhill for result in results],
                dir_name=[result.dir_name for result in results],
                tags=tags,
                **task_document_kwargs,
            )

    @classmethod
    def to_mol_or_struct_metadata_doc(
        cls,
        ase_calculator_name: str,
        result: AseResult|list[AseResult],
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



        if isinstance(task_doc.mol_or_struct, Structure) or isinstance(task_doc.mol_or_struct[0], Structure):
            meta_class = AseStructureTaskDoc
            k = "structure"
            if relax_cell := getattr(task_doc, "relax_cell", None):
                kwargs.update({"relax_cell": relax_cell})
        elif isinstance(task_doc.mol_or_struct, Molecule) or isinstance(task_doc.mol_or_struct[0], Molecule):
            meta_class = AseMoleculeTaskDoc
            k = "molecule"

        kwargs.update({k: task_doc.mol_or_struct, f"meta_{k}": task_doc.mol_or_struct})

        return getattr(meta_class, f"from_{k}")(**kwargs)






