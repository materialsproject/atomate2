from atomate2.common.schemas.magnetism import MagnetismOutput as MagnetismOutputBase


class MagnetismOutput(MagnetismOutputBase):
    """
    Defines the output structure for a magnetic ordering calculation. This is
    implemented here for VASP. See base class for more details.
    """

    @classmethod
    def from_task_document(cls, task_document):
        output = task_document["output"]
        dir_name = task_document["dir_name"]
        structure = output["structure"]
        magmoms = output["magmom"]

        analyzer = CollinearMagneticStructureAnalyzer(input_structure, threshold=0.61)

        return cls(dir_name=dir_name, structure=structure, magmoms=magmoms)
