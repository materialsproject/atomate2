import numpy as np
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer

from atomate2.common.schemas.magnetism import MagneticOrderingInput
from atomate2.common.schemas.magnetism import (
    MagneticOrderingOutput as MagneticOrderingOutputBase,
)
from atomate2.common.schemas.magnetism import (
    MagneticOrderingRelaxation as MagneticOrderingRelaxationBase,
)


class MagneticOrderingRelaxation(MagneticOrderingRelaxationBase):
    """
    Defines the relaxation output document for a magnetic ordering calculation.

    The construction of this document is implemented here for VASP. See base class for
    more details.
    """

    @classmethod
    def from_task_document(cls, task_document, uuid=None):
        """
        Construct a MagneticOrderingRelaxation from a task document. This does not include
        the uuid, which must be set separately.
        """
        dir_name = task_document.dir_name
        structure = task_document.structure
        input_structure = task_document.input.structure
        energy = task_document.output.energy
        energy_per_atom = task_document.output.energy_per_atom
        symmetry = structure.get_space_group_info()[0]

        input_analyzer = CollinearMagneticStructureAnalyzer(
            input_structure, threshold=0.61
        )
        analyzer = CollinearMagneticStructureAnalyzer(structure, threshold=0.61)
        ordering = analyzer.ordering

        input_magmoms = input_analyzer.magmoms
        magmoms = analyzer.magmoms

        input_order_check = [0 if abs(m) < 0.61 else m for m in input_magmoms]
        order_check = [0 if abs(m) < 0.61 else m for m in magmoms]

        ordering_changed = not np.array_equal(
            np.sign(input_order_check), np.sign(order_check)
        )
        input_symmetry = input_structure.get_space_group_info()[0]
        symmetry_changed = symmetry != input_symmetry

        input = MagneticOrderingInput(
            structure=input_structure,
            ordering=input_analyzer.ordering,
            symmetry=input_symmetry,
        )

        return cls(
            uuid=uuid,
            dir_name=dir_name,
            input=input,
            structure=structure,
            symmetry_changed=symmetry_changed,
            ordering_changed=ordering_changed,
            ordering=ordering,
            symmetry=symmetry,
            energy=energy,
            energy_per_atom=energy_per_atom,
        )


class MagneticOrderingOutput(MagneticOrderingOutputBase):
    """
    Defines the static output for a magnetic ordering calculation.

    The construction of this document is implemented here for VASP. See base class for
    more details.
    """

    @classmethod
    def from_task_document(cls, task_document, uuid=None):
        """
        Construct a MagneticOrderingOutput from a task document. This does not include
        the uuid, which must be set separately.
        """
        dir_name = task_document.dir_name
        structure = task_document.structure
        input_structure = task_document.input.structure
        energy = task_document.output.energy
        energy_per_atom = task_document.output.energy_per_atom
        symmetry = structure.get_space_group_info()[0]

        input_analyzer = CollinearMagneticStructureAnalyzer(
            input_structure, threshold=0.61
        )
        analyzer = CollinearMagneticStructureAnalyzer(structure, threshold=0.61)
        ordering = analyzer.ordering

        input_magmoms = input_analyzer.magmoms
        magmoms = analyzer.magmoms

        input_order_check = [0 if abs(m) < 0.61 else m for m in input_magmoms]
        order_check = [0 if abs(m) < 0.61 else m for m in magmoms]

        ordering_changed = not np.array_equal(
            np.sign(input_order_check), np.sign(order_check)
        )
        input_symmetry = input_structure.get_space_group_info()[0]
        symmetry_changed = symmetry != input_symmetry

        total_magnetization = analyzer.total_magmoms
        num_formula_units = structure.composition.get_reduced_composition_and_factor()[
            1
        ]
        total_magnetization_per_formula_unit = total_magnetization / num_formula_units
        total_magnetization_per_unit_volume = total_magnetization / structure.volume

        input = MagneticOrderingInput(
            structure=input_structure,
            ordering=input_analyzer.ordering,
            symmetry=input_symmetry,
        )

        return cls(
            uuid=uuid,
            dir_name=dir_name,
            input=input,
            structure=structure,
            ordering=ordering,
            magmoms=list(magmoms),
            symmetry=symmetry,
            energy=energy,
            energy_per_atom=energy_per_atom,
            total_magnetization=total_magnetization,
            total_magnetization_per_formula_unit=total_magnetization_per_formula_unit,
            total_magnetization_per_unit_volume=total_magnetization_per_unit_volume,
            ordering_changed=ordering_changed,
            symmetry_changed=symmetry_changed,
        )
