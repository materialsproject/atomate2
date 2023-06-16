from atomate2.common.schemas.magnetism import MagnetismOutput as MagnetismOutputBase


class MagnetismOutput(MagnetismOutputBase):
    """
    Defines the output structure for a magnetic ordering calculation. This is
    implemented here for VASP. See base class for more details.
    """

    def from_task_document(self, task_document):
        output = task_document["output"]
        self.dir_name = task_document["dir_name"]
        self.structure = output["structure"]
        self.magmoms = output["magmom"]
