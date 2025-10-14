"""Define schemas for SQS runs."""

from pydantic import BaseModel, Field

try:
    from emmet.core.types.enums import ValueEnum
except ImportError:
    from emmet.core.utils import ValueEnum

from pymatgen.core import Structure
from pymatgen.transformations.transformation_abc import AbstractTransformation


class SQSMethod(ValueEnum):
    """Define possible SQS methods used."""

    MCSQS = "mcsqs"
    ICET_ENUM = "icet-enumeration"
    ICET_MCSQS = "icet-monte_carlo"


class TransformTask(BaseModel):
    """Schematize a transformation run."""

    transformation: AbstractTransformation = Field(
        description="The transformation applied to a structure."
    )

    final_strcture: Structure = Field(
        description="The structure after the transformation."
    )

    input_structure: Structure = Field(
        description="The structure before the transformation."
    )


class SQSTask(TransformTask):
    """Structure the output of SQS runs."""

    sqs_method: SQSMethod | None = Field(None, description="The SQS protocol used.")
    final_objective: float | None = Field(
        None,
        description=(
            "The minimum value of the SQS obejective function, "
            "corresponding to the structure in `final_structure`"
        ),
    )
    sqs_structures: list[Structure] | None = Field(
        None, description="A list of other good SQS candidates."
    )
    sqs_scores: list[Structure] | None = Field(
        None,
        description=(
            "The objective function values for the structures in `sqs_structures`"
        ),
    )
